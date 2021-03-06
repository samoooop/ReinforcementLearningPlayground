import gym
import gym_rle
import time
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout
import numpy as np
import time

def prep(img):
    # down sample image
    img = img[30:-60,:,]
    img = img[::2,::2,]
    import skimage.color as skc
    img = skc.rgb2grey(img)
    img = np.expand_dims(img, axis = 2)
    # print(img.shape)
    return img


def getModel(action_count):
    model = Sequential()
    model.add(Convolution2D(16, (16,16), strides = 2,
                activation='relu', input_shape=(80,80,1)))
    model.add(Convolution2D(16, (8,8), strides = 2, 
                activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(action_count))
    return model

def getTrainData(histories, expected_rewards, reward):
    x = np.array(histories)
    # replace maximum reward with actual reward(maximum = action selected)
    expected_rewards = np.array(expected_rewards)
    # print('xxx',expected_rewards.shape)
    maxi = np.argmax(expected_rewards, axis = 1)
    expected_rewards[:,maxi] = reward
    # print('xxx',expected_rewards.shape)
    return x, expected_rewards
    
load = True
env_name = 'AirRaid-v0'
env = gym.make(env_name)
obs = env.reset()
# env.render()

if not load:
    model = getModel(env.action_space.n)
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])
else:
    print('loading model')
    model = load_model(env_name + '.h5')

stime = time.time()
print(env.action_space)
total_step = 0
total_reward = 0

histories = []
# actions = []
rewards = []
last_reward = np.zeros(10,dtype = np.float)
i = 0
j = 0
while True:
    x = prep(obs)
    x = np.array([x])
    expected_reward = model.predict([x])[0]
    # print(expectedReward)
    action = np.argmax(expected_reward)
    # print(action)
    if np.random.rand() < 0.1:
        expected_reward[action] = 1000000 # TODO
        action = env.action_space.sample()
    # print(expected_reward)

    obs, reward, done, info = env.step(action)
    # print(reward)
    # env.render()
    # time.sleep(0.05)

    rewards.append(expected_reward)
    histories.append(x[0])
    # actions.append(action)

    total_step += 1
    total_reward += reward
    if done:
        last_reward[j] = total_reward
        j = (j+1) % 10
        i += 1
        if i%20 == 0:
            print('Saveing at',i)
            model.save(env_name + '.h5')
        # print('drop',i)
        # fit NN
        x,y = getTrainData(histories, rewards, total_reward)
        model.fit(x,y,batch_size = 32, epochs = 10, verbose = 0)
        histories = []
        rewards = []

        elapsedTime = time.time() - stime
        stime = time.time()
        print('Episode done took ', elapsedTime
                ,'total step', total_step
                ,'total reward', total_reward)
        print('Mean reward(last 10) :',last_reward.mean(),'with var :',last_reward.var())
        total_step = 0
        total_reward = 0
        obs = env.reset()

