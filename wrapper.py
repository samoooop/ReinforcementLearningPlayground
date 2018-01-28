import gym

class SurviveEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return 1
