import argparse
import pathlib
from .core import RLAlg
from .config import Config
import dataclasses


def parse_args():
    parser = argparse.ArgumentParser(
        description='Additional arguments that match config fields name can be passed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--load', type=pathlib.Path, help='path to the experiment dir')
    for field in dataclasses.fields(Config):
        parser.add_argument(f'--{field.name}', type=field.type, default=field.default, help=str(field.type))
    return parser.parse_args()


def make_config(args):
    config = Config()
    fields = {k: v for k, v in vars(args).items() if k in vars(config).keys()}
    return dataclasses.replace(config, **fields)


def train():
    args = parse_args()
    config = make_config(args)

    if args.load:
        config = config.load(args.load / 'config.yml')

    alg = RLAlg(config)

    if args.load:
        alg.load(args.load)
    alg.learn()


if __name__ == "__main__":
    train()
