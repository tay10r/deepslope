from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import torch
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid

from loguru import logger

import numpy as np

from PIL import Image

from deepslope.config import Config, get_config
from deepslope.nn.refinement import RefinementNet
from deepslope.data.dataset import RefinementDataset


class Program:
    def __init__(self, config_path):
        self.__config: Config = get_config(config_path)
        self.__net: RefinementNet | None = None
        self.__dataset: RefinementDataset | None = None
        Path(self.__config.tmp_path).mkdir(exist_ok=True)
        self.__tmp_path = Path(self.__config.tmp_path) / uuid4().hex
        self.__tmp_path.mkdir(exist_ok=False)
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

        num_epochs = self.__config.num_epochs
        for i in range(num_epochs):
            epoch_loss = self.__run_training_epoch(loader)
            self.__run_test()
            self.__epoch += 1
            logger.info(f'Epoch [{i}/{num_epochs}]: {epoch_loss}')

        self.__dataset.close()
        self.__save_net()

    def __run_training_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        self.__net.train()
        epoch_loss = 0.0
        init_step = self.__step
        for sample in loader:

            tile, expected, _ = sample
            tile = tile.to(self.__device)
            expected = expected.to(self.__device)

            self.__optimizer.zero_grad()
            result = self.__net(tile)
            spatial_loss = mse_loss(result, expected)
            step_loss = spatial_loss
            epoch_loss += step_loss.item()
            step_loss.backward()
            self.__optimizer.step()

            if self.__config.log_dataset:
                g = make_grid(tile.cpu(), nrow=4)
                g_name = f'input_{self.__step:08}.png'
                save_image(g, self.__get_tmp_path(g_name))

                g = make_grid(expected.cpu(), nrow=4)
                g_name = f'expected_{self.__step:08}.png'
                save_image(g, self.__get_tmp_path(g_name))

            self.__step += 1

        num_steps = self.__step - init_step
        avg_epoch_loss = epoch_loss / num_steps
        return avg_epoch_loss

    def __run_test(self):
        self.__net.eval()
        with torch.no_grad():
            test_image = Image.open(self.__config.test_image)
            test_image = torch.Tensor(np.array(test_image))
            scale = self.__config.test_height / 256.0
            test_image = test_image.to(torch.float32) * scale
            test_image = test_image.to(self.__device)
            test_image = torch.unsqueeze(test_image, dim=0)
            test_image = torch.unsqueeze(test_image, dim=0)
            result: torch.Tensor = self.__net(test_image)
            result_path = self.__tmp_path / f'test_{self.__epoch:04}.png'
            result = result.cpu()
            result = result.squeeze(dim=0)
            result = result.squeeze(dim=0)
            result = np.clip(result.numpy() / scale, 0, 255).astype(np.uint8)
            img = Image.fromarray(result)
            img.save(result_path)

    def __save_net(self):
        assert self.__net is not None
        model_path = str(self.__tmp_path / 'refinement_net.pt')
        torch.save(self.__net.state_dict(), model_path)

    def __open_dataset(self):
        self.__dataset = RefinementDataset(self.__config.dems,
                                           self.__config.frequency_cutoffs,
                                           self.__config.sample_size,
                                           self.__config.seed,
                                           self.__config.normalize_height)

    def __open_net(self):
        model_path = self.__tmp_path / 'model.pt'
        self.__net = RefinementNet()
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
