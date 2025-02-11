from deepslope.monitoring.monitor import Monitor

import numpy as np

import vz


class VZMonitor(Monitor):
    def __init__(self):
        self.window = vz.Window()
        self.epoch = 0

    def close(self):
        self.window.close()

    def begin_epoch(self, epoch: int):
        self.epoch = epoch

    def end_epoch(self):
        self.epoch += 1

    def log_loss(self, name: str, loss: float):
        self.window.log_loss(name, float(self.epoch), loss)

    def log_grad(self, layer_name: str, grad: float):
        self.window.log_grad(layer_name, grad)

    def log_test_result(self, image: np.ndarray):
        self.window.log_test_results(image)
