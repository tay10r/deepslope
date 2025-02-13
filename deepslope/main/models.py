from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn

from deepslope.nn.diffusion import TerrainDiffusion64


@dataclass
class ModelInfo:
    input_size: int


class _ModelFactory(ABC):
    @abstractmethod
    def make(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def info(self) -> ModelInfo:
        raise NotImplementedError()


class _DiffuseModelFactory(_ModelFactory):
    def __init__(self, res: int):
        self.res = res

    def make(self) -> nn.Module:
        if self.res == 64:
            return TerrainDiffusion64()
        raise ValueError(f'Not a supported resolution: {self.res}')

    def info(self) -> ModelInfo:
        return ModelInfo(input_size=self.res)


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, _ModelFactory] = {
            'terrain-diffuse-64': _DiffuseModelFactory(64)
        }

    def info(self, name: str) -> ModelInfo:
        return self.models[name].info()

    def list(self) -> list[str]:
        names = []
        for name in self.models.keys():
            names.append(name)
        return name

    def get(self, name) -> nn.Module:
        return self.models[name].make()
