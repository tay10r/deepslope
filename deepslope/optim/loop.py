from abc import ABC, abstractmethod
import signal
import multiprocessing as mp
import multiprocessing.synchronize as sync
import time

import numpy as np

from loguru import logger

import vz

from deepslope.monitoring import CompoundMonitor, VZMonitor
from deepslope.config import Config
from deepslope.state import GlobalState

stop_flag = False


def signal_handler(signum: int, frame):
    global stop_flag
    stop_flag = True
    logger.info(f'Caught signal {signum}, exiting.')


class TaskFactory(ABC):
    @abstractmethod
    def create_task(queue: mp.Queue, stop_event: sync.Event, config: Config, state: GlobalState) -> mp.Process:
        raise NotImplementedError()


class Loop:
    """
    This is a base class for a training loop.
    """

    def __init__(self,  task_factory: TaskFactory, config: Config, state: GlobalState):
        self.config = config
        self.state = state
        self.monitor = CompoundMonitor()
        self.task_factory = task_factory

    def run(self):
        if self.config.enable_vz:
            logger.info('Starting visualization module.')
            vz.init()
            self.monitor.add(VZMonitor())

        global stop_flag
        stop_flag = False
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info(
            'Interrupts (SIGINT, SIGTERM) are setup. You may use Ctrl+C to stop.')

        info_queue = mp.Queue()
        stop_event = mp.Event()
        process: mp.Process = self.task_factory.create_task(
            info_queue, stop_event, self.config, self.state)

        process.start()

        while not stop_flag:
            while not info_queue.empty():
                info = info_queue.get_nowait()
                self.__process_info(info)
            if self.config.enable_vz:
                vz.poll_events()
            # no need to poll more than we really have to.
            time.sleep(0.05)

        stop_event.set()

        process.join(timeout=5.0)
        if process.is_alive():
            logger.error(
                'Training process did not join, forcefully terminating it.')
            process.terminate()
        else:
            logger.info('Training process exited gracefully.')

        info_queue.close()

        self.monitor.close()

        if self.config.enable_vz:
            logger.info('Closing visualization module.')
            vz.teardown()

    def __process_epoch_start(self, epoch_info: dict):
        epoch = int(epoch_info['epoch'])
        self.monitor.begin_epoch(epoch)
        logger.info(f'Starting epoch {epoch}')

    def __process_epoch_finish(self, epoch_info: dict):
        self.monitor.end_epoch()

    def __process_grad(self, grad: dict):
        for name, value in grad.items():
            self.monitor.log_grad(name, value)

    def __process_loss(self, loss_info: dict):
        for name, value in loss_info.items():
            self.monitor.log_loss(name, value)
            logger.info(f'Loss "{name}": {value}')

    def __process_test_result(self, test_result: np.ndarray):
        self.monitor.log_test_result(test_result)

    def __process_error(self, error_info: dict):
        global stop_flag
        stop_flag = True
        logger.error(error_info['description'])

    def __process_info(self, info: dict):
        if 'epoch_start' in info:
            self.__process_epoch_start(info['epoch_start'])
        if 'epoch_finish' in info:
            self.__process_epoch_finish(info['epoch_finish'])
        if 'test_result' in info:
            self.__process_test_result(info['test_result'])
        if 'grad' in info:
            self.__process_grad(info['grad'])
        if 'loss' in info:
            self.__process_loss(info['loss'])
        if 'error' in info:
            self.__process_error(info['error'])
