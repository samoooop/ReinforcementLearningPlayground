import gym
import gym_rle
from wrapper import SurviveEnv

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = gym.make('GradiusIii-ram-v0')
    env = bench.Monitor(env, logger.get_dir())
    # from baselines.common.atari_wrappers import wrap_deepmind
    # env = wrap_deepmind(env, episode_life = False, clip_rewards = False)
    # env = SurviveEnv(env)
    model = deepq.models.mlp(
        hiddens=[256],
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=50000,
        exploration_fraction=0.8,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.9,
        prioritized_replay=bool(args.prioritized)
    )
    act.save("gradius_model.pkl") 
    env.close()


if __name__ == '__main__':
    main()
