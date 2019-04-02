import argparse


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, choices=['dqn', 'reinforce', 'reinforce_multi', 'ppo', 'ppo_multi'], help='algorithm')
parser.add_argument('--env', help='environment name', type=str, default='CartPole-v0')
parser.add_argument('--eval', help='evaluate (instead of train)', action='store_true')
args = parser.parse_args()

print(f'Running {args.env} with {args.algo}.')
if args.algo == 'dqn':
    from dqn.runners import train as train_runner
    from dqn.runners import evaluate
elif args.algo == 'reinforce':
    from reinforce.runners import train as train_runner
    from reinforce.runners import evaluate
elif args.algo == 'reinforce_multi':
    from reinforce.runners import train_multi as train_runner
    from reinforce.runners import evaluate
elif args.algo == 'ppo':
    from ppo.runners import train as train_runner
    from ppo.runners import evaluate
elif args.algo == 'ppo_multi':
    from ppo.runners import train_multi as train_runner
    from ppo.runners import evaluate

if args.eval:
    evaluate(args.env)
else:
    train_runner(args.env)
