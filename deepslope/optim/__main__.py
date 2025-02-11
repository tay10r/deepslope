from pathlib import Path
from argparse import ArgumentParser

from loguru import logger

from deepslope.optim.loop import Loop, TaskFactory
from deepslope.optim.tasks.train_diffusion import TrainDiffusionTaskFactory
from deepslope.config import Config, get_config
from deepslope.state import GlobalState, load_global_state, store_global_state


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--task', type=str, default='train_diffusion')
    args = arg_parser.parse_args()
    task: str = args.task

    task_factory: TaskFactory | None = None
    match task:
        case 'train_diffusion':
            task_factory = TrainDiffusionTaskFactory()

    if task_factory is None:
        logger.error(f'Unknown task {task}.')
        return

    config = get_config()
    state = load_global_state(Path(config.tmp_path) / 'state.json')
    loop = Loop(task_factory, config, state)
    loop.run()


if __name__ == '__main__':
    main()
