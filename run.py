#!/usr/bin/env python3

import argparse
from common.atari import is_atari

parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, choices=['dqn', 'reinforce', 'reinforce_multi', 'ppo', 'ppo_multi'], help='algorithm')
parser.add_argument('--env', help='environment name', type=str, default='CartPole-v0')
parser.add_argument('--eval', help='evaluate (instead of train)', action='store_true')
args = parser.parse_args()

print(f'Running {args.env} with {args.algo}.')
if args.algo == 'dqn':
    if is_atari(args.env):
        from dqn.runners_atari import train
        from dqn.runners_atari import evaluate
    else:
        from dqn.runners_lowdim import train
        from dqn.runners_lowdim import evaluate

elif args.algo == 'reinforce':
    if is_atari(args.env):
        from reinforce.runners_atari import train
        from reinforce.runners_atari import evaluate
    else:
        from reinforce.runners_lowdim import train
        from reinforce.runners_lowdim import evaluate

elif args.algo == 'reinforce_multi':
    if is_atari(args.env):
        from reinforce.runners_atari import train_multi as train
        from reinforce.runners_atari import evaluate
    else:
        from reinforce.runners import train_multi as train
        from reinforce.runners import evaluate

elif args.algo == 'ppo':
    if is_atari(args.env):
        from ppo.runners_atari import train
        from ppo.runners_atari import evaluate
    else:
        from ppo.runners import train
        from ppo.runners import evaluate
        
elif args.algo == 'ppo_multi':
    if is_atari(args.env):
        from ppo.runners_atari import train_multi as train
        from ppo.runners_atari import evaluate
    else:
        from ppo.runners import train_multi as train
        from ppo.runners import evaluate

if args.eval:
    evaluate(args.env)
else:
    train(args.env)
