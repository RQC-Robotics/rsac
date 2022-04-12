import argparse
import pathlib
from .core import RLAlg, Config
import dataclasses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=pathlib.Path, help='path to the experiment with config and weights')
    parser.add_argument('--task', type=str, default='walker_stand')
    parser.add_argument('--observe', type=str, choices=['states', 'pixels', 'point_cloud'], default='point_cloud')
    parser.add_argument('--aux_loss', type=str, choices=['None', 'contrastive', 'reconstruction'], default='None')
    parser.add_argument('--logdir', type=str, default='logdir')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def make_config(args):
    config = Config()
    return dataclasses.replace(config, **vars(args))


if __name__ == "__main__":
    args = parse_args()
    config = make_config(args)
    
    if args.load:
        config = config.load(args.load / 'config.yml')
    
    alg = RLAlg(config)
    
    if args.load:
        alg.load(args.load)
    alg.learn()
