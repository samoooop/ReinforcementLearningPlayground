import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind

from baselines import deepq


def main():
    env = gym.make('GradiusIii-v0')
    env = wrap_deepmind(env, episode_life = False, clip_rewards = False)
    act = deepq.load("result/survive.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)



if __name__ == '__main__':
    main()
