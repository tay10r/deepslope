from dataclasses import fields
from pathlib import Path

from loguru import logger

from deepslope.main.config import Config
from deepslope.main.state import State


def interactive_override_dataclass(config: Config) -> Config:
    for field in fields(config):
        current_value = getattr(config, field.name)
        user_input = input(f'{field.name} (default: {current_value}): ').strip()
        if user_input:
            field_type = field.type
            if field_type == int:
                user_input = int(user_input)
            elif field_type == float:
                user_input = float(user_input)
            setattr(config, field.name, user_input)
    return config


def new_config(state: State, args):
    config = Config()
    name: str = ''
    while True:
        name = input('Enter a name for this config: ').strip()
        if name == '':
            logger.error('Config name must not be empty.')
        elif name in state.configs:
            logger.error('Config name must be unique.')
        else:
            break
    config.tmp_path = str(Path(config.tmp_path) / name)
    interactive_override_dataclass(config)
    logger.info(f'Generating new config "{name}".')
    for field in fields(config):
        value = getattr(config, field.name)
        logger.info(f'{field.name} = {value}')
    state.configs[name] = config
    logger.info(f'Set current config to "{name}"')
    state.current_config = name
    logger.info(f'Generating path for configuration at: {config.tmp_path}')
    Path(config.tmp_path).mkdir(parents=True, exist_ok=False)
