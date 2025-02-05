from abc import ABC, abstractmethod

import numpy as np
import rasterio
import rasterio.windows


class ElevationModel(ABC):
    """
    Represents an abstract elevation model.
    It is meant to facilitate reading elevation data from disk
    without putting the entire model into memory.
    """

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def get_size(self) -> tuple[int, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        raise NotImplementedError()


class TiffElevationModel(ElevationModel):
    def __init__(self, path: str):
        self.data = rasterio.open(path)

    def close(self):
        self.data.close()

    def get_size(self) -> tuple[int, int]:
        return (self.data.width, self.data.height)

    def get_tile(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        window = rasterio.windows.Window.from_slices((y, y + h), (x, x + w))
        elevation = self.data.read(1, window=window)
        return elevation
