#!/usr/bin/env python3

import argparse
from common.atari import is_atari


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, choices=['dqn', 'reinforce', 'ppo'], help='algorithm')
parser.add_argument('--multi', help='use multiple parallel environments', action='store_true')
parser.add_argument('--env', help='environment name', type=str, default='CartPole-v0')
parser.add_argument('--eval', help='evaluate (instead of train)', action='store_true')
args = parser.parse_args()

print(f'Running {args.env} with {args.algo}.')

# import selected modules
if is_atari(args.env):  # load atari runners
    if args.multi:
        exec(f'from {args.algo}.runners_atari import train_multi as train')
    else:
        exec(f'from {args.algo}.runners_atari import train')
    exec(f'from {args.algo}.runners_atari import evaluate')
else:  # load lowdim runners
    if args.multi:
        exec(f'from {args.algo}.runners_lowdim import train_multi as train')
    else:
        exec(f'from {args.algo}.runners_lowdim import train')
    exec(f'from {args.algo}.runners_lowdim import evaluate')

# execute
if args.eval:
    evaluate(args.env)
else:
    train(args.env)
