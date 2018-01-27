import gym

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return 1
    