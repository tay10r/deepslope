from dataclasses import dataclass, asdict
import json
from pathlib import Path

import torch


@dataclass
class Config:
    # The list of DEMs to use for training data.
    # This are normally geotiff files.
    dems: list[str]
    # The terrains go through several stages of low-pass filtering,
    # in order to simulate level of erosion and change. This is the
    # list of cutt of frequencies and should be sorted lowest to highest
    # and range from 0 to 1.
    frequency_cutoffs: list[float]
    # When training, use this sample size as input to and output from the network.
    sample_size: tuple[int, int]
    # How many samples to batch during each training iterationo.
    batch_size: int = 16
    # How much to modify the weights from each gradient after back prop
    learning_rate: float = 0.0001
    # The device to run the network on.
    # If this is not specified, it will default to CUDA if CUDA is available and fallback to the CPU.
    device: str | None = None
    # The seed for initializing random number generators.
    seed: int = 0
    # The path to save data to during training.
    tmp_path: str = 'tmp'
    # A path to an image that will be used for testing and logging progress.
    test_image: str = 'test/data/input_1.png'
    # If set to true, the network input and target images will be logged.
    # This can be useful for debugging augmentations to the input data.
    log_dataset: bool = False
    # Whether or not to normalize height.
    normalize_height: bool = False
    # The maximum height of the test image is this many meters.
    test_height: float = 50.0


def get_config(path: str = 'config.json') -> Config:
    """
    Gets a configuration file by either reading an existing one or creating a new one.
    """
    if Path(path).exists():
        return open_config(path)
    new_config = Config(dems=[],
                        frequency_cutoffs=[0.005, 0.01, 0.02, 0.04, 1.0],
                        sample_size=(256, 256))
    if torch.cuda.is_available():
        new_config.device = 'cuda'
    else:
        new_config.device = 'cpu'
    with open(path, 'w') as f:
        json.dump(asdict(new_config), f, indent=2)
    return new_config


def open_config(path: str):
    """
    Opens a configuration file.
    """
    with open(path, 'r') as f:
        return Config(**json.load(f))
