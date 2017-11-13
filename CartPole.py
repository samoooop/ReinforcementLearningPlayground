import gym
import time
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import pickle
import skimage.color as skc

def prep(img):
    # down sample image
    # print(img.shape)
    img = img[40:-10,:,]
    img = img[::2,::2,]
    img = skc.rgb2grey(img)
    img = np.expand_dims(img, axis = 2)
    
    imgs[0:3] = imgs[-3:]
    imgs[3] = img
    if total_step >= 4:
        img = imgs[-4:].mean(axis = 0)
    # print(img.shape)
    # print(img)
    # f = open('img.pkl', 'wb')
    # pickle.dump(img, f, pickle.HIGHEST_PROTOCOL)
    # plt.imshow(img.reshape(80,80))
    # plt.imshow()
    # time.sleep(1)
    return img


def getModel(action_count):
    model = Sequential()
    # model.add(Convolution2D(16, (8,8), strides = 4,
                # activation='relu', input_shape=(80,80,1)))
    # model.add(Convolution2D(8, (4,4), strides = 2, 
                # activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(16, activation='relu', init = 'uniform', input_shape = (2,4,)))
    model.add(Flatten())
    model.add(Dense(8, activation = 'relu', init = 'uniform'))
    # model.add(Dropout(0.1))
    model.add(Dense(action_count))
    return model

def getTrainData(histories, rewards, actions, dones, new_states):
    expected_rewards = model.predict(histories)
    expected_rewards_from_next_step = model.predict(new_states)
    expected_rewards_from_next_step = np.max(expected_rewards_from_next_step, axis = 1)
    # print('z')
    # actions_index = np.array(actions)
    not_dones = np.logical_not(dones)
    # print(dones,type(not_dones),type(done))
    # print(not_dones,expected_rewards_from_next_step.shape,rewards.shape)
    # print(expected_rewards_from_next_step.shape,rewards.shape)
    expected_rewards[dones, actions[dones]] = rewards[dones]
    expected_rewards[not_dones, actions[not_dones]] = (rewards[not_dones] 
                                                +  l * expected_rewards_from_next_step[not_dones])
    # print('xxx',expected_rewards.shape)
    # maxi = np.argmax(expected_rewards, axis = 1)
    # expected_rewards[:,maxi] = reward
    # print('xxx',expected_rewards.shape)
    return expected_rewards
    
load = True
env_name = 'CartPole-v0'
env = gym.make(env_name)
obs = env.reset()
# env.render()

if not load:
    model = getModel(env.action_space.n)
    model.compile(optimizer='sgd',
                loss='mean_squared_error',
                metrics=[])
else:
    print('loading model')
    model = load_model(env_name + '.h5')

stime = time.time()
print(env.action_space)
print(env.observation_space)
total_step = 0
total_reward = 0

memory = deque()
memory_size = 100
histories = []
actions = []
rewards = []
dones = []
new_states = []
action_picked = np.zeros(6)
last_reward = np.zeros(10,dtype = np.float)
i = 0
j = 0
e = 0.0
decay_rate = 1.0
l = 0.9
learning_rate = 0.2
x = np.stack((obs, obs), axis = 0)
print(np.array([x]))
state = np.array([x])
print('state',state.shape)
while True:
    # x = np.array([obs.flatten()])
    expected_reward = model.predict(state)[0]
    action = np.argmax(expected_reward)
    action_picked[action] += 1
    if np.random.rand() < e:
        action = env.action_space.sample()

    # print(expected_reward,action)
    obs, reward, done, info = env.step(action)

    # print(state.shape, np.array([obs]).shape)
    new_state = np.concatenate((state[0][1:], np.array([obs])), axis = 0)
    new_state = np.array([new_state])

    # if done:
    #     expected_reward[action] = reward
    # else:
    #     # xp = np.array([prep(obs)])
    #     # xp = np.array([obs.flatten()])
    #     expected_reward_from_next_step = np.max(model.predict(new_state)[0])
    #     expected_reward[action] = reward + l*expected_reward_from_next_step

    # if i%20 == 0:
    # env.render()

    actions.append(action)
    rewards.append(reward)
    histories.append(state[0])
    dones.append(done)
    new_states.append(new_state[0])
    # actions.append(action)

    state = new_state
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
        a = np.array(actions)
        d = np.array(dones)
        ns = np.array(new_states)
        # print('x')
        if x.shape[0] >= 1000:
            choices = np.random.choice(x.shape[0],1000)
            x = x[choices]
            y = y[choices]
            a = a[choices]
            d = d[choices]
            ns = ns[choices]
        # print('y')
        y = getTrainData(x, y, a, d, ns)

        print('training with x:',x.shape,' y:',y.shape)
            # x,y = getTrainData(histories, rewards, total_reward)
        model.fit(x,y,batch_size = 64, epochs = 10, verbose = 0)
        score = model.evaluate(x, y, verbose = 0)
        print('show some prediction')
        choices = np.random.choice(x.shape[0], 5)
        # xc = x[choices]
        # yc = y[choices]
        # ypc = model.predict(xc)
        # for k in range(0,5):
            # print(yc[k]-ypc[k])
        print('Done training ->',score,model.metrics_names)
        # pred_y = model.predict(x[0:5])
        # for i in range(0,5):
            # print(pred_y[i],y[i])

        elapsedTime = time.time() - stime
        stime = time.time()
        print('action picked:', action_picked / action_picked.sum())
        print('Episode %d done took ' % i, elapsedTime ,' sec',' e:',e)
        print('total ', total_step,' step','total reward','\033[93m'+str(total_reward)+'\033[0m')
        print('Mean reward(last 10) :',last_reward.mean(),'with var :',last_reward.var())
        print('-------------------------------------------------------------------------')
        total_step = 0
        total_reward = 0
        e = e * decay_rate
        # histories = []
        # rewards = []
        # actions = []
        action_picked = np.zeros(6)
        obs = env.reset()
        if i > 3000:
            e = 0.1

