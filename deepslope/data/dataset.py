from random import Random

from torch import unsqueeze, Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

from PIL import Image

import numpy as np

from deepslope.data.filter import Filter
from deepslope.data.elevation import ElevationModel, TiffElevationModel


class TiffDataset(Dataset):
    def __init__(self,
                 dem_path_list: list,
                 frequency_cutoffs: list[float],
                 sample_size: tuple[int, int],
                 seed: int,
                 normalize: bool):
        """
        Constructs a new dataset instance.

        :dem_path_list: The list of file paths for each DEM included in the dataset.
        :frequency_cutoffs: The levels of frequency cutoffs
        """
        self.models: list[ElevationModel] = []
        for dem_path in dem_path_list:
            self.models.append(TiffElevationModel(dem_path))
        self.filters = []
        for cutoff in frequency_cutoffs:
            self.filters.append(Filter(cutoff, normalize_output=normalize))
        self.sample_size = sample_size
        self.rng = Random(seed)
        self.normalize = normalize

    def close(self):
        for m in self.models:
            m.close()

    def __len__(self):
        # The length is practically unlimited, but PyTorch excepts a length regardless.
        # This is mostly an arbitrary number.
        return 1024

    def __getitem__(self, _):
        # Note that the index is disregarded because this dataset is mostly procedural.
        # Every time this function is called, a unique sample is generated.
        model = self.models[self.rng.randint(0, len(self.models) - 1)]
        size = model.get_size()
        x = self.rng.randint(0, size[0] - self.sample_size[0])
        y = self.rng.randint(0, size[1] - self.sample_size[1])
        tile = model.get_tile(x, y, self.sample_size[0], self.sample_size[1])
        tile_img = Image.fromarray(tile)

        if self.rng.randint(0, 1) == 1:
            tile_img = F.horizontal_flip(tile_img)
        if self.rng.randint(0, 1) == 1:
            tile_img = F.vertical_flip(tile_img)
        tile = np.array(tile_img)

        if self.normalize:
            # min max normalization
            min_h = np.min(tile)
            max_h = np.max(tile)
            tile = (tile - min_h) / (max_h - min_h)
        else:
            # we still need to start at zero, but we keep the scale of the features
            min_h = np.min(tile)
            tile = tile - min_h

        filter = self.filters[self.rng.randint(0, len(self.filters) - 1)]
        expected_fft, low_freq_tile = filter(tile)
        tile = Tensor(tile)
        low_freq_tile = Tensor(low_freq_tile)
        expected_fft = Tensor(expected_fft)
        return unsqueeze(low_freq_tile, dim=0), unsqueeze(tile, dim=0), unsqueeze(expected_fft, dim=0)
