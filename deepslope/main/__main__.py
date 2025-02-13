from argparse import ArgumentParser
from pathlib import Path
import json
from dataclasses import asdict

from loguru import logger

from deepslope.main.new_config import new_config
from deepslope.main.new_dataset import new_dataset
from deepslope.main.export import export
from deepslope.main.train import train
from deepslope.main.state import State


def main():
    state = State()
    state_path = Path('state.json')
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = State(**json.load(f))
    else:
        logger.info('Generating new state.json file on exit.')

    parser = ArgumentParser(description="Deep learning for terrain modeling.")
    subparsers = parser.add_subparsers(dest='command')

    new_config_parser = subparsers.add_parser('new-config', help='Generate a new configuration.')
    new_config_parser.set_defaults(func=new_config)

    new_dataset_parser = subparsers.add_parser('new-dataset', help='Generates a new dataset.')
    new_dataset_parser.add_argument(
        'filename', nargs='*', help='The list of source files to sample from (such as TIFF files)')
    new_dataset_parser.add_argument('--seed', type=int, default=0, help='The seed used to initialize the PRNG.')
    new_dataset_parser.add_argument('--procedural', action='store_true',
                                    help='Whether or not to generate a procedural dataset.')
    new_dataset_parser.add_argument('--num-train-samples', type=int, default=100000,
                                    help='The number of training samples to produce.')
    new_dataset_parser.add_argument('--num-test-samples', type=int, default=10000,
                                    help='The number of test samples to produce.')
    new_dataset_parser.set_defaults(func=new_dataset)

    train_parser = subparsers.add_parser('train', help='Trains a model in the current configuration.')
    train_parser.add_argument('--epochs', type=int, default=16)
    train_parser.add_argument('--log-input', action='store_true')
    train_parser.set_defaults(func=train)

    export_parser = subparsers.add_parser('export')
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    if args.command:
        args.func(state, args)
    else:
        parser.print_help()

    with open(state_path, 'w') as f:
        json.dump(asdict(state), f, indent=2)


if __name__ == '__main__':
    main()
