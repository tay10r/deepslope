from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid

from loguru import logger

import numpy as np

from PIL import Image

from deepslope.config import Config, get_config
from deepslope.nn.net import Net
from deepslope.data.dataset import TiffDataset


class Program:
    def __init__(self, config_path):
        self.__config = get_config(config_path)
        self.__net: Net | None = None
        self.__dataset: TiffDataset | None = None
        Path(self.__config.tmp_path).mkdir(exist_ok=True)
        self.__optimizer: torch.optim.Optimizer | None = None
        self.__device = torch.device(self.__config.device)
        self.__epoch = 0
        self.__step = 0

    def run(self):
        if len(self.__config.dems) == 0:
            logger.error(
                'There needs to be at least one DEM to pull data from.')
            return
        self.__open_net()
        self.__open_dataset()

        loader = torch.utils.data.DataLoader(
            self.__dataset, batch_size=self.__config.batch_size)

        self.__net = self.__net.to(self.__device)

        num_epochs = 32
        for i in range(num_epochs):
            logger.info(f'Epoch [{i}/{num_epochs}]')
            self.__run_training_epoch(loader)
            logger.info('Running test.')
            self.__run_test()
            self.__epoch += 1

        self.__dataset.close()
        self.__save_net()

    def __run_training_epoch(self, loader: torch.utils.data.DataLoader):
        self.__net.train()
        for sample in loader:
            tile, expected = sample
            tile = tile.to(self.__device)
            expected = expected.to(self.__device)
            self.__optimizer.zero_grad()
            result = self.__net(tile)
            loss = mse_loss(result, expected)
            logger.info(f'Loss: {loss.item()}')
            loss.backward()
            self.__optimizer.step()
            if self.__config.log_dataset:
                save_image(make_grid(tile.cpu(), nrow=4),
                           self.__get_tmp_path(f'input_{self.__step:08}.png'))
                save_image(make_grid(expected.cpu(), nrow=4),
                           self.__get_tmp_path(f'expected_{self.__step:08}.png'))
            self.__step += 1

    def __run_test(self):
        self.__net.eval()
        with torch.no_grad():
            test_image = Image.open(self.__config.test_image)
            test_image = torch.Tensor(np.array(test_image))
            test_image = test_image.to(torch.float32) / 256.0
            test_image = test_image.to(self.__device)
            test_image = torch.unsqueeze(test_image, dim=0)
            test_image = torch.unsqueeze(test_image, dim=0)
            result: torch.Tensor = self.__net(test_image)
            result_path = self.__get_tmp_path(f'test_{self.__epoch:04}.png')
            result = result.cpu()
            result = result.squeeze(dim=0)
            result = result.squeeze(dim=0)
            img = Image.fromarray((result.numpy() * 255).astype(np.uint8))
            img.save(result_path)

    def __get_tmp_path(self, name: str) -> str:
        return str(Path(self.__config.tmp_path) / name)

    def __save_net(self):
        assert self.__net is not None
        model_path = self.__get_tmp_path('model.pt')
        torch.save(self.__net.state_dict(), model_path)

    def __open_dataset(self):
        self.__dataset = TiffDataset(
            self.__config.dems, self.__config.frequency_cutoffs, self.__config.sample_size, self.__config.seed)

    def __open_net(self):
        model_path = self.__get_tmp_path('model.pt')
        self.__net = Net()
        if Path(model_path).exists():
            self.__net.load_state_dict(
                torch.load(model_path, weights_only=True))
            logger.info(f'Loaded model "{model_path}"')
        else:
            logger.info('Creating new model.')
        self.__optimizer = torch.optim.SGD(
            self.__net.parameters(), self.__config.learning_rate)


def main():
    parser = ArgumentParser(
        prog='train', description='For training a model to add detail to terrains.')
    parser.add_argument('--config', type=str, default='config.json',
                        help='The path of the config file to use.')
    args = parser.parse_args()

    program = Program(args.config)
    program.run()


if __name__ == '__main__':
    main()
