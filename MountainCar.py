import gym
import gym_rle
import time
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout
import numpy as np
import time
from collections import deque


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
    # model.add(Convolution2D(16, (8,8), strides = 4,
    #             activation='relu', input_shape=(80,80,1)))
    # model.add(Convolution2D(16, (4,4), strides = 2, 
    #             activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(64, activation='relu',input_shape=[2]))
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
    
load = False
env_name = 'MountainCar-v0'
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

memory = deque()
memory_size = 100
histories = []
# actions = []
rewards = []
last_reward = np.zeros(10,dtype = np.float)
i = 0
j = 0
e = 0.0
decay_rate = 0.99
l = 0.5
while True:
    x = obs
    # x = prep(obs)
    # x = np.array([x])
    expected_reward = model.predict([x])[0]
    action = np.argmax(expected_reward)
    if np.random.rand() < e:
        action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    if done:
        expected_reward[action] = reward
    else:
        xp = np.array([prep(obs)])
        expected_reward_from_next_step = np.max(model.predict([xp])[0])
        expected_reward[action] = reward + l*expected_reward_from_next_step
    # print(reward)
    env.render()
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
            print('\033[93m'+'Saveing at',i)
            print('\033[0m')
            model.save(env_name + '.h5')
        
        # fit NN
        x = np.array(histories)
        y = np.array(rewards)
        # if x.shape[0] >= 1000:
            # choices = np.random.choice(x.shape[0],1000)
            # x = x[choices]
            # y = y[choices]
        print('training with x:',x.shape,' y:',y.shape)
            # x,y = getTrainData(histories, rewards, total_reward)
        model.fit(x,y,batch_size = 32, epochs = 50, verbose = 0)
        score = model.evaluate(x, y, verbose = 0)
        print('Done training ->',score,model.metrics_names)
        # pred_y = model.predict(x[0:5])
        # for i in range(0,5):
            # print(pred_y[i],y[i])

        elapsedTime = time.time() - stime
        stime = time.time()
        print('Episode %d done took ' % i, elapsedTime ,' sec',' e:',e)
        print('total ', total_step,' step','total reward','\033[93m'+str(total_reward)+'\033[0m')
        print('Mean reward(last 10) :',last_reward.mean(),'with var :',last_reward.var())
        print('-------------------------------------------------------------------------')
        total_step = 0
        total_reward = 0
        e = e * decay_rate
        histories = []
        rewards = []
        obs = env.reset()

