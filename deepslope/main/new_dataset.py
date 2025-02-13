import sys
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

import pg

import numpy as np

from loguru import logger

from deepslope.main.state import State
from deepslope.main.config import Config
from deepslope.main.models import ModelRegistry, ModelInfo


class _Sampler(ABC):
    @abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError()


class _ProceduralSampler(_Sampler):
    def __init__(self, seed: int, size: int):
        self.generator = pg.Generator(seed=seed)
        self.size = size

    def sample(self) -> np.ndarray:
        return self.generator.make_noise(self.size, self.size)


def generate_folder(num_samples: int, sampler: _Sampler, out_path: Path):
    for i in range(num_samples):
        img_path = str(out_path / f'{i:06}.png')
        logger.info(f'{i}/{num_samples}: Generating "{img_path}"')
        tile = sampler.sample()
        tile_min = np.min(tile)
        tile_max = np.max(tile)
        if tile_min != tile_max:
            tile = (tile - tile_min) / (tile_max - tile_min)
        img = Image.fromarray((tile * 65535.0).astype(np.uint16), mode='I;16')
        img.save(img_path)


def generate_dataset(num_train_samples: int, num_test_samples: int, sampler: _Sampler, out_path: Path):
    train_path = out_path / 'train'
    train_path.mkdir()
    test_path = out_path / 'test'
    test_path.mkdir()
    generate_folder(num_train_samples, sampler, train_path)
    generate_folder(num_test_samples, sampler, test_path)


def new_dataset(state: State, args):
    name: str | None = None
    while True:
        name = input('Enter the dataset name: ')
        if name == '':
            logger.error('The name cannot be empty.')
        elif name in state.datasets:
            logger.error('The name must be unique.')
        else:
            break

    if state.current_config is None:
        logger.error('No config currently setup.')
        sys.exit(1)

    if not state.current_config in state.configs:
        logger.error(f'Config "{state.current_config}" is missing.')

    config: Config = Config(**state.configs[state.current_config])

    registry = ModelRegistry()

    if not config.model_name in registry.list():
        logger.error(f'Model "{config.model_name}" does not exist.')
        sys.exit(1)

    model_info: ModelInfo = registry.info(config.model_name)

    sampler: _Sampler | None = None

    if len(args.filename) == 0:
        if args.procedural:
            sampler = _ProceduralSampler(seed=args.seed, size=model_info.input_size)
    else:
        if args.procedural:
            logger.error('Cannot use --procedural in addition to filenames.')
            sys.exit(1)

    if sampler is None:
        logger.error('Need either at least one filename (or set the --procedural flag)')
        sys.exit(1)

    path = Path('datasets') / name
    path.mkdir(parents=True, exist_ok=False)

    generate_dataset(num_train_samples=args.num_train_samples,
                     num_test_samples=args.num_test_samples,
                     sampler=sampler,
                     out_path=path)
    state.datasets[name] = {
        'path': str(path)
    }
    logger.info(f'Setting current dataset to "{name}"')
    state.current_dataset = name
