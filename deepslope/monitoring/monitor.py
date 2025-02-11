from abc import ABC, abstractmethod

import numpy as np


class Monitor(ABC):
    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def begin_epoch(self, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def end_epoch(self):
        raise NotImplementedError()

    @abstractmethod
    def log_loss(self, name: str, value: float):
        raise NotImplementedError()

    @abstractmethod
    def log_grad(self, weight_name: str, value: float):
        raise NotImplementedError()

    @abstractmethod
    def log_test_result(self, image: np.ndarray):
        raise NotImplementedError()


class CompoundMonitor(Monitor):
    def __init__(self):
        self.monitors: list[Monitor] = []

    def add(self, monitor: Monitor):
        self.monitors.append(monitor)

    def close(self):
        for m in self.monitors:
            m.close()

    def begin_epoch(self, epoch: int):
        for m in self.monitors:
            m.begin_epoch(epoch)

    def end_epoch(self):
        for m in self.monitors:
            m.end_epoch()

    def log_loss(self, name: str, value: float):
        for m in self.monitors:
            m.log_loss(name, value)

    def log_grad(self, weight_name: str, value: float):
        for m in self.monitors:
            m.log_grad(weight_name, value)

    def log_test_result(self, image: np.ndarray):
        for m in self.monitors:
            m.log_test_result(image)
