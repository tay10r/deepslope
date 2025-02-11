from random import Random
from pathlib import Path

from torch import unsqueeze, Tensor
from torch.utils.data import Dataset

from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as transforms

from PIL import Image

import numpy as np

from deepslope.data.filter import Filter
from deepslope.data.elevation import ElevationModel, TiffElevationModel, FakeElevationModel


class DiffusionDataset(Dataset):
    def __init__(self, root: Path, batch_size: int, transform: transforms.Transform | None = None):
        self.samples: list[Tensor] = []
        for s in Path(root).glob('*.png'):
            img = Image.open(str(s))
            tensor = Tensor(np.array(img))
            self.samples.append(tensor)
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self) -> int:
        l = len(self.samples)
        l = int(l / self.batch_size) * self.batch_size
        return l

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        s = self.samples[idx].float() * (1.0 / 255.0)
        if self.transform is not None:
            s = self.transform(s)
        return unsqueeze(s, dim=0)


class GANPreparedDataset(Dataset):
    """
    If you preprocess the data for training the GAN, you can use this dataset.
    It speeds up the training process by avoiding all the procedural generation
    and resizing, as well as storing the entire dataset in RAM.
    """

    def __init__(self, root: Path, batch_size: int, transform: transforms.Transform | None = None):
        self.real: list[Tensor] = []
        for s in (root / '0').glob('*.png'):
            img = Image.open(str(s))
            tensor = Tensor(np.array(img))
            self.real.append(tensor)
        self.fake: list[Tensor] = []
        for s in (root / '1').glob('*.png'):
            img = Image.open(str(s))
            tensor = Tensor(np.array(img))
            self.fake.append(tensor)
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self) -> int:
        l = min(len(self.real), len(self.fake))
        l = int(l / self.batch_size) * self.batch_size
        return l

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        r = self.real[idx].float() * (1.0 / 255.0)
        f = self.fake[idx].float() * (1.0 / 255.0)
        if self.transform is not None:
            r = self.transform(r)
            f = self.transform(f)
        return unsqueeze(r, dim=0), unsqueeze(f, dim=0)


class GANDataset(Dataset):
    def __init__(self,
                 dem_path_list: list[str],
                 sample_size: tuple[int, int],
                 seed: int):
        self.real_models: list[ElevationModel] = []
        for dem_path in dem_path_list:
            self.real_models.append(TiffElevationModel(dem_path))
        self.sample_size = sample_size
        self.rng = Random(seed)
        self.simplex_model = FakeElevationModel(seed=seed)

    def close(self):
        for model in self.real_models:
            model.close()

    def __len__(self) -> int:
        return 1024

    def __getitem__(self, _) -> tuple[Tensor, Tensor]:
        """
        :returns: A real sample for the discriminator, an artificial input for the discriminator, an artificial input for the generator.
        """
        d_real = self.__get_real_tile()
        g_fake = self.__get_fake_tile()
        return d_real, g_fake

    def __get_real_tile(self) -> Tensor:
        model = self.real_models[self.rng.randint(
            0, len(self.real_models) - 1)]
        return self.__get_tile(model)

    def __get_fake_tile(self) -> Tensor:
        return self.__get_tile(self.simplex_model)

    def __get_tile(self, model: ElevationModel):
        size = model.get_size()
        x = self.rng.randint(0, size[0] - self.sample_size)
        y = self.rng.randint(0, size[1] - self.sample_size)
        tile = model.get_tile(x, y, self.sample_size, self.sample_size)
        tile_img = Image.fromarray(tile)

        if self.rng.randint(0, 1) == 1:
            tile_img = F.horizontal_flip(tile_img)
        if self.rng.randint(0, 1) == 1:
            tile_img = F.vertical_flip(tile_img)

        tile = np.array(tile_img)
        min_h = np.min(tile)
        tile = tile - min_h
        return unsqueeze(Tensor(tile), dim=0)


class RefinementDataset(Dataset):
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
        :sample_size: The height and width of each sample.
        :seed: The seed that initializes the random number generator.
        :normalize: Whether or not to normalize the height map.
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
        # This is mostly an arbitrary number, since the TIFF files are generally very
        # large and the dataset randomly crops from them.
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
