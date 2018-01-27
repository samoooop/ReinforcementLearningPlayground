import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind

from baselines import deepq


def main():
    env = gym.make('GradiusIii-v0')
    env = wrap_deepmind(env, episode_life = False, clip_rewards = False)
    # act = deepq.load("result/50mstep.py")

    for _ in range(0, 100):
        obs, done = env.reset(), False
        episode_rew = 0
        i = 0
        while not done:
            env.render()
            # obs, rew, done, _ = env.step(0)
            for _ in range(0, 4):
                obs, rew, done, _ = env.step(0)
                print(obs.shape)
                episode_rew += rew
            i = i+1
        print("Episode reward", episode_rew, i)


if __name__ == '__main__':
    main()
