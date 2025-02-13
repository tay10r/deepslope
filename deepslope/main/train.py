from pathlib import Path
from random import Random
import math

from loguru import logger

from tqdm import tqdm

from torch.optim import Adam
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch

from torchvision.transforms import v2 as transforms
from torchvision.utils import make_grid, save_image

from deepslope.main.state import State
from deepslope.main.models import ModelRegistry
from deepslope.main.config import Config
from deepslope.data.dataset import DiffusionDataset


def __forward_diffusion(image: Tensor, noise: Tensor, num_steps: int, alpha_bars: list[Tensor]) -> Tensor:
    batch_size = image.shape[0]

    timesteps = torch.randint(0, num_steps, (batch_size,), device=image.device)

    alpha_bar = torch.tensor([alpha_bars[t] for t in timesteps], device=image.device).view(batch_size, 1, 1, 1)

    noisy_images = torch.sqrt(alpha_bar) * image + torch.sqrt(1 - alpha_bar) * noise

    return noisy_images


def train(state: State, args):
    log_input: bool = args.log_input
    config = Config(**state.configs[state.current_config])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = state.datasets[state.current_dataset]['path']
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    def load_callback(i, total): return logger.info(f'Loading dataset {int(i*100/total)}%')

    logger.info('Loading training dataset.')
    train_ds = DiffusionDataset(Path(dataset_path) / 'train', config.batch_size, transform=train_transform)
    train_ds.load(load_callback)

    logger.info('Loading test dataset.')
    test_ds = DiffusionDataset(Path(dataset_path) / 'test', config.batch_size)
    test_ds.load(load_callback)

    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, config.batch_size, shuffle=False)
    models = ModelRegistry()
    model = models.get(config.model_name).to(device)
    optimizer = Adam(params=model.parameters(), lr=config.learning_rate)

    epoch = 0
    best_test_loss = 1.0e9

    train_state_path = Path(config.tmp_path) / 'checkpoint.pt'
    best_model_path = Path(config.tmp_path) / 'best.pt'
    if train_state_path.exists():
        checkpoint = torch.load(train_state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_test_loss = checkpoint['best_test_loss']

    betas = torch.linspace(config.beta_min, config.beta_max,
                           config.diffusion_steps, device=device,
                           requires_grad=False)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    epochs: int = args.epochs

    for i in range(0, epochs):
        logger.info(f'-- Epoch {epoch} [{i}/{epochs}] --')
        model.train()

        counter = 0
        train_tq = tqdm(train_loader)
        ema_train_loss: None | float = None
        train_loss_sum = 0.0
        for sample in train_tq:
            sample = sample.to(device)
            noise = torch.randn_like(sample, device=device)
            noisy_sample = __forward_diffusion(sample, noise, config.diffusion_steps, alpha_bars)
            if log_input:
                g = make_grid(noisy_sample)
                save_image(g, str(Path(config.tmp_path) / f'train_input_{epoch:04}_{counter:06}.png'))
            optimizer.zero_grad()
            predicted_noise = model(noisy_sample)
            loss = F.mse_loss(predicted_noise, noise)
            train_loss = loss.item()
            train_loss_sum += train_loss
            ema_train_loss = train_loss if ema_train_loss is None else 0.01 * train_loss + 0.99 * ema_train_loss
            loss.backward()
            optimizer.step()
            counter += 1
            train_tq.set_description(f'Epoch {epoch} train loss: {ema_train_loss:.04}')
        train_loss_avg = train_loss_sum / len(train_loader)

        test_loss_sum = 0.0
        ema_test_loss: float | None = None
        counter = 0
        test_tq = tqdm(test_loader)
        with torch.no_grad():
            model.eval()
            for sample in test_tq:
                sample = sample.to(device)
                noise = torch.randn_like(sample, device=device)
                noisy_sample = __forward_diffusion(sample, noise, config.diffusion_steps, alpha_bars)
                predicted_noise = model(noisy_sample)
                loss = F.mse_loss(predicted_noise, noise)
                test_loss = loss.item()
                test_loss_sum += test_loss
                ema_test_loss = test_loss if ema_test_loss is None else 0.01 * test_loss + 0.99 * ema_test_loss
                test_tq.set_description(f'Epoch {epoch} test loss: {ema_test_loss:.04}')
                counter += 1
        test_loss_avg = test_loss_sum / len(test_loader)
        logger.info(f'Epoch {epoch} complete. Train loss: {train_loss_avg:.04}, Test loss: {test_loss_avg:.04}')
        if test_loss_avg < best_test_loss:
            logger.info('New best model.')
            torch.save(model.state_dict(), str(best_model_path))
            best_test_loss = test_loss_avg

        epoch += 1
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_test_loss': best_test_loss
        }
        torch.save(checkpoint, str(train_state_path))
