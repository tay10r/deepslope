import multiprocessing as mp
import multiprocessing.synchronize as sync
from pathlib import Path
from random import Random

import numpy as np

from torch.utils.data import DataLoader
from torch import Tensor, unsqueeze, concat
import torch
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from torchvision.transforms import v2 as transforms
from torchvision.utils import make_grid

from deepslope.optim.loop import TaskFactory
from deepslope.config import Config
from deepslope.state import GlobalState
from deepslope.data.dataset import DiffusionDataset
from deepslope.nn.diffusion import DiffusionModel
from deepslope.monitoring.grad import compute_avg_grad


class _Task:
    def __init__(self, info_queue: mp.Queue, stop_event: sync.Event, config: Config, state: GlobalState):
        self.info_queue = info_queue
        self.stop_event = stop_event
        self.should_stop = False
        self.config = config
        self.state = state
        self.rng = Random(self.config.seed)
        self.device = torch.device(self.config.device)
        self.betas = torch.linspace(1.0e-4, self.config.diffusion_noise,
                                    self.config.diffusion_steps, device=self.device,
                                    requires_grad=False)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        batch_size = self.config.batch_size
        data_path = Path(self.config.tmp_path) / 'data'
        self.train_dataset = DiffusionDataset(str(data_path / 'train'),
                                              batch_size,
                                              transform=train_transform)
        self.test_dataset = DiffusionDataset(str(data_path / 'test'),
                                             batch_size)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size, shuffle=True)
        self.model = DiffusionModel().to(self.device)
        self.optimizer = SGD(self.model.parameters(),
                             self.config.learning_rate)
        self.step_lr = StepLR(self.optimizer, step_size=100, gamma=0.1)

    def run(self):
        epoch = 0
        while not self.stop_event.is_set() and not self.should_stop:
            self.__publish_epoch_begin(epoch)
            train_data: tuple[float, dict[str, float]] | None = self.__train()
            if train_data is None:
                # interrupted by stop event
                break
            train_loss, grad_avg = train_data
            self.__publish_loss('train_loss', train_loss)
            self.__publish_grad(grad_avg)
            with torch.no_grad():
                self.__test()
            self.__publish_epoch_end()
            epoch += 1

    def __forward_diffusion(self, image: Tensor, noise: Tensor, step: int) -> Tensor:
        alpha_bar = self.alpha_bars[step].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * image + torch.sqrt(1.0 - alpha_bar) * noise

    def __test(self):
        self.model.eval()
        img = self.test_dataset[0].to(self.device)
        noise = torch.randn_like(img, device=self.device)
        s = int(self.config.sample_size / self.config.reduce)
        results = Tensor(size=(0, 1, s, s)).to(self.device)
        sample = self.__forward_diffusion(img, noise, step=-1)
        results = concat((results, unsqueeze(img, dim=0)), dim=0)
        results = concat((results, sample), dim=0)
        for i in reversed(range(1, self.config.diffusion_steps)):
            if self.stop_event.is_set():
                return None
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            predicted_noise = self.model(sample)
            x_mean = (1 / torch.sqrt(alpha)) * (
                sample - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
            )
            if i > 1:
                noise = torch.randn_like(sample)
                sigma = torch.sqrt(beta)
                sample = x_mean + sigma * noise
            else:
                sample = x_mean
        results = concat((results, sample), dim=0)
        self.__publish_test_result(
            make_grid(results, nrow=1).cpu()[0].numpy())

    def __train(self) -> tuple[float, dict[str, float]] | None:
        self.model.train()
        train_loss = 0.0
        steps = self.config.diffusion_steps
        grad_avg: dict[str, float] | None = None
        for sample in self.train_loader:
            if self.stop_event.is_set():
                return None
            sample = sample.to(self.device)
            i = self.rng.randint(0, steps - 1)
            noise = torch.randn_like(sample, device=self.device)
            noisy_sample = self.__forward_diffusion(sample, noise, i)
            self.optimizer.zero_grad()
            predicted_noise = self.model(noisy_sample)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            self.optimizer.step()
            g = compute_avg_grad(self.model)
            if grad_avg is None:
                grad_avg = g
            else:
                for name, val in g.items():
                    grad_avg[name] += val
            train_loss += loss.item()
        n = len(self.train_loader)
        if grad_avg is not None:
            for name, val in grad_avg.items():
                grad_avg[name] = val / n
        train_loss /= n
        self.step_lr.step()
        return train_loss, grad_avg

    def __publish_epoch_end(self):
        msg = {'epoch_finish': {}}
        self.info_queue.put(msg)

    def __publish_epoch_begin(self, epoch: int):
        msg = {
            'epoch_start': {
                'epoch': epoch
            }
        }
        self.info_queue.put(msg)

    def __publish_test_result(self, result: np.ndarray):
        msg = {
            'test_result': result
        }
        self.info_queue.put(msg)

    def __publish_grad(self, grad: dict[str, float]):
        msg = {
            'grad': grad
        }
        self.info_queue.put(msg)

    def __publish_error(self, what: str):
        msg = {
            'error': {
                'description': what
            }
        }
        self.info_queue.put(msg)
        self.should_stop = True

    def __publish_loss(self, name: str, value: float):
        msg = {
            'loss': {
                name: value
            }
        }
        self.info_queue.put(msg)


def train(info_queue: mp.Queue, stop_event: sync.Event, config: Config, state: GlobalState):
    error: str | None = None
    try:
        task = _Task(info_queue, stop_event, config, state)
        task.run()
    except Exception as e:
        error = str(e)
    if error is not None:
        info_queue.put({'error': {'description': error}})


class TrainDiffusionTaskFactory(TaskFactory):
    def create_task(self, info_queue: mp.Queue, stop_event: sync.Event, config: Config, state: GlobalState) -> mp.Process:
        return mp.Process(target=train, args=(info_queue, stop_event, config, state))
