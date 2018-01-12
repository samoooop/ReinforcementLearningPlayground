import gym
import gym_rle
import numpy as np

env = gym.make('GradiusIii-v0')
# env.mode = 'human'
obs = env.reset()
print(obs.shape)
assert False
env.render()
done = False

isGoingUp = np.ones(131072, dtype = np.bool)
lastObs = np.zeros(131072)
i = 0
while not done:
    # action = env.action_space.random()
    action = 13
    i = i+1
    obs, reward, done, info = env.step(action)
    if i < 200:
        isGoingUp = isGoingUp & (obs < lastObs)
        a = (obs > lastObs)
        lastObs = obs
        for j in range(0, a.shape[0]):
            if a[j] == True and j < 1000 :
                # print(j)
                pass
    env.render()
    print(obs[10])
    # print(i)

obs = env.reset()
env.render()
done = False

isGoingUpNOOP = np.ones(131072, dtype = np.bool)
lastObs = np.zeros(131072)
i = 0
while not done:
    # action = env.action_space.random()
    action = 0
    i = i+1
    obs, reward, done, info = env.step(action)
    if i < 200:
        isGoingUpNOOP = isGoingUpNOOP & (obs >= lastObs)
        lastObs = obs
    env.render()



print('xxx')
for i in range(0,isGoingUp.shape[0]):
    if isGoingUp[i] == 1 and isGoingUpNOOP[i] != isGoingUp[i]:
        print(i)
        # pass