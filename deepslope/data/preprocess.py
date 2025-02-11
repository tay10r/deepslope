"""
This module builds a dataset to train the model on.
It helps using pre-computed data since there's a lot of procedural
generation and resizing done to the original TIFF data.
"""

from pathlib import Path

import numpy as np

from PIL import Image

from loguru import logger

from torch import Tensor, squeeze

from deepslope.config import Config, get_config
from deepslope.data.dataset import GANDataset


class Program:
    def __init__(self):
        self.config: Config = get_config()
        self.out_path = Path(self.config.tmp_path) / 'data'
        self.train_path = self.out_path / 'train'
        self.test_path = self.out_path / 'test'
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.train_path.mkdir(exist_ok=True)
        self.test_path.mkdir(exist_ok=True)

    def run(self):
        if len(self.config.dems) == 0:
            logger.error('Need at least one DEM to build a dataset.')
            return
        dataset = GANDataset(
            self.config.dems, self.config.sample_size, seed=self.config.seed)
        self.__generate(num_samples=1024, dataset=dataset,
                        suffix=self.train_path)
        self.__generate(num_samples=self.config.batch_size,
                        dataset=dataset, suffix=self.test_path)

    def __generate(self, num_samples: int, dataset: GANDataset, suffix: Path):
        n = 0
        while n < num_samples:
            for sample in dataset:
                real, fake = sample
                # The fake sample was for a previous experiment type.
                # We can ignore it for now, and eventually refactor it out.
                self.__save_image(real, suffix / f'{n:04}.png')
                n += 1
                if n >= num_samples:
                    break

    def __save_image(self, image: Tensor, path: Path):
        image = squeeze(image, dim=0)
        array: np.ndarray = image.numpy()
        min_h = np.min(array)
        max_h = np.max(array)
        range = max_h - min_h
        array: np.ndarray = (array - min_h) / range
        img = Image.fromarray((array * 255.0).astype(np.uint8))
        s = int(self.config.sample_size / self.config.reduce)
        img = img.resize((s, s))
        img.save(str(path))


def main():
    program = Program()
    program.run()


if __name__ == '__main__':
    main()
