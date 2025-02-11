"""
This module trains a GAN network, which can take an initial terrain
and noise and produce a new, more realistic terrain.
"""

from random import Random
from pathlib import Path

import numpy as np

from loguru import logger

from PIL import Image

from torchvision.utils import save_image, make_grid
from torchvision.transforms import v2 as transforms

from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

import vz

from deepslope.config import Config, GANConfig, get_config
from deepslope.state import GlobalState, load_global_state, store_global_state
from deepslope.nn.discriminator import Discriminator
from deepslope.nn.generator import Generator
from deepslope.data.dataset import GANPreparedDataset as Dataset
from deepslope.monitoring.monitor import Monitor, CompoundMonitor
from deepslope.monitoring.vz_monitor import VZMonitor
from deepslope.monitoring.grad import visit_grad

NOISE_SIZE = 256 + 32


class Program:
    def __init__(self, config: Config, state: GlobalState):
        self.config = config
        self.rng = Random(config.seed)
        self.device = torch.device(config.device)
        self.discriminator = Discriminator().to(self.device)
        self.generator = Generator().to(self.device)
        data_path = Path(self.config.tmp_path) / 'data'
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        self.train_dataset = Dataset(root=(data_path / 'train'),
                                     batch_size=self.config.batch_size,
                                     transform=train_transform)
        self.test_dataset = Dataset(root=(data_path / 'test'),
                                    batch_size=self.config.batch_size)
        self.loader = DataLoader(self.train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True)
        self.tmp_path = Path(self.config.tmp_path) / \
            'experiments' / str(state.num_experiments)
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.step = 0
        state.num_experiments += 1
        self.monitors = CompoundMonitor()
        self.monitors.add(VZMonitor())

    def close(self):
        pass

    def run(self):

        gan_config = GANConfig(**self.config.gan)
        disc_optimizer = SGD(self.discriminator.parameters(),
                             gan_config.discriminator_learning_rate)
        gen_optimizer = SGD(self.generator.parameters(),
                            self.config.learning_rate)

        epochs = self.config.num_epochs
        logger.info('Begining training loop.')
        last_dacc = 0.0
        max_dacc = 0.7
        for i in range(epochs):
            dloss = 0.0
            dacc = 0.0
            gloss = 0.0
            self.monitors.begin_epoch(epoch=i)
            for sample in self.loader:

                if self.config.enable_vz:
                    vz.poll_events()

                real, fake = sample
                if self.config.log_dataset:
                    self.__log_data(real, 'real')
                    self.__log_data(fake, 'fake')

                self.generator.train()
                fake = fake.to(self.device)
                noise = torch.rand(self.config.batch_size,
                                   1, NOISE_SIZE, NOISE_SIZE)
                noise = noise.to(self.device)
                generated: Tensor = self.generator(fake, noise)
                disc_real_loss, disc_real_acc = self.__train_disc(real, 1.0)
                disc_fake_loss, disc_fake_acc = self.__train_disc(
                    generated.detach(), 0.0)
                dacc += (disc_real_acc + disc_fake_acc) * 0.5

                # Train Discriminator (if the accuracy is below the threshold)
                if last_dacc < max_dacc:
                    disc_optimizer.zero_grad()
                    disc_loss: Tensor = disc_real_loss + disc_fake_loss
                    dloss += disc_loss.item()
                    disc_loss.backward()
                    disc_optimizer.step()
                else:
                    disc_loss = 0.0

                # Train Generator
                gen_optimizer.zero_grad()
                g_pred = self.discriminator(generated)
                s = self.config.batch_size
                expected = torch.ones(size=(s, 1))
                expected = expected.to(self.device)
                g_loss_1 = F.binary_cross_entropy_with_logits(g_pred, expected)
                # g_loss_2 = F.l1_loss(generated, fake) * 0.2
                # g_loss = g_loss_1 + g_loss_2
                g_loss = g_loss_1
                g_loss.backward()
                gloss += g_loss.item()
                gen_optimizer.step()
                visit_grad(self.generator, self.monitors)

                self.step += 1

            # Compute average loss
            n = len(self.loader)
            dloss /= n
            dacc /= n
            gloss /= n
            self.monitors.log_loss('generator', gloss)
            self.monitors.log_loss('discriminator', dloss)
            self.monitors.end_epoch()
            last_dacc = dacc

            # Log results
            with torch.no_grad():
                self.__run_test(i)
            logger.info(f'[{i}/{epochs}]: D:{dloss:.3}|{dacc:.2} G:{gloss:.3}')

    def __run_test(self, epoch: int):
        self.generator.eval()
        test_image = Image.open(self.config.test_image)
        test_image = test_image.resize((256, 256))
        test_image = torch.Tensor(np.array(test_image))
        scale = self.config.test_height / 255.0
        test_image = test_image.to(torch.float32) * scale
        test_image = test_image.to(self.device)
        test_image = torch.unsqueeze(test_image, dim=0)
        test_image = torch.unsqueeze(test_image, dim=0)
        noise = torch.rand(1, 1, NOISE_SIZE, NOISE_SIZE).to(self.device)
        result: Tensor = self.generator(test_image, noise)
        result = result.cpu()
        result = result.squeeze(dim=0)
        result = result.squeeze(dim=0)
        result = np.clip(result.numpy() / scale, 0, 255).astype(np.uint8)
        img = Image.fromarray(result)
        img.save(self.tmp_path / f'test_{epoch:04}.png')

    def __log_data(self, x: Tensor, name: str):
        path = str(self.tmp_path / (f'{name}_{self.step:04}.png'))
        g = make_grid(x, nrow=4)
        save_image(g, path)

    def __train_disc(self, tile: Tensor, expected_value: float) -> tuple[Tensor, float]:
        tile = tile.to(self.device)
        pred = self.discriminator(tile)
        s = self.config.batch_size
        expected = torch.full(size=(s, 1), fill_value=expected_value)
        expected = expected.to(self.device)
        accuracy = ((F.sigmoid(pred) >= 0.5) == expected).float().sum() / s
        return F.binary_cross_entropy_with_logits(pred, expected), accuracy


def main():
    config: Config = get_config()
    state: GlobalState = load_global_state(
        Path(config.tmp_path) / 'state.json')
    if len(config.dems) == 0:
        logger.error('Need at least one real DEM to train this model.')
        return
    if config.enable_vz:
        vz.init()
    program = Program(config, state)
    store_global_state(Path(config.tmp_path) / 'state.json', state)
    program.run()
    program.close()
    store_global_state(Path(config.tmp_path) / 'state.json', state)
    if config.enable_vz:
        vz.teardown()


if __name__ == '__main__':
    main()
