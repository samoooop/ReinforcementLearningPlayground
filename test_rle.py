import gym
import gym_rle
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout
import numpy as np
import time  
import skimage.color as skc

def prep(img):
    # down sample image
    img = img[::2,::2,]
    img = img[12:,14:-14,]
    img = skc.rgb2grey(img)
    img = np.expand_dims(img, axis = 2)
    # print(img.shape)
    plt.imshow(img.reshape(80, 80))
    plt.show()
    return img


def get_model():
    model = Sequential()
    model.add(Convolution2D(16, (16,16), strides = 2,
                activation='relu', input_shape=(100,100,1)))
    model.add(Convolution2D(16, (8,8), strides = 2, 
                activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
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
    

model = get_model()
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

env = gym.make('ClassicKong-v0')
obs = env.reset()
env.render()

stime = time.time()
print(env.action_space)
total_step = 0
total_reward = 0

histories = []
# actions = []
rewards = []
i = 0
j = 0
while True:
    x = prep(obs)
    x = np.array([x])
    expected_reward = model.predict([x])[0]
    # print(expectedReward)
    action = np.argmax(expected_reward)
    # print(action)
    if np.random.rand(1)[0] < 0.3:
        expected_reward[action] = 1000000 # TODO
        action = env.action_space.sample()
    # print(expected_reward)

    obs, reward, done, info = env.step(action)
    # print(reward)
    env.render()
    # time.sleep(0.2)

    rewards.append(expected_reward)
    histories.append(x[0])
    # actions.append(action)

    total_step += 1
    total_reward += reward
    if done:
        print('drop',i)
        # fit NN
        x,y = getTrainData(histories, rewards, total_reward)
        model.fit(x,y,batch_size = 32, epochs = 10, verbose = 0)
        histories = []
        rewards = []

        elapsedTime = time.time() - stime
        stime = time.time()
        print('Episode done took ', elapsedTime, total_step, total_reward)
        total_step = 0
        total_reward = 0
        obs = env.reset()

