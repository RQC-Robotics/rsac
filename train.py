import argparse
from src.core import RLAlg, Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config')
    parser.add_argument('--load', type=bool, default=False)
    return parser.parse_args()


args = parse_args()
config = Config()
try:
    config.load(args.config)
except FileNotFoundError:
    print('FileNotFound. Example config created')
    config.save(args.config)

alg = RLAlg(config)
if args.load:
    alg.load()
alg.learn()
