import argparse


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, choices=['dqn', 'reinforce', 'reinforce_multi', 'ppo', 'ppo_multi'], help='algorithm')
parser.add_argument('--env', help='environment name', type=str, default='CartPole-v0')
parser.add_argument('--eval', help='evaluate (instead of train)', action='store_true')
args = parser.parse_args()

print(f'Running {args.env} with {args.algo}')
if args.algo == 'dqn': import dqn.run as run
elif args.algo == 'reinforce': import reinforce.run as run
elif args.algo == 'reinforce_multi': import reinforce.run_multi as run
elif args.algo == 'ppo': import ppo.run as run
elif args.algo == 'ppo_multi': import ppo.run_multi as run

if args.eval:
    run.evaluate(args.env)
else:
    run.train(args.env)
