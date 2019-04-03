from __future__ import print_function
import gym
import numpy as np
import time

env = gym.make("FrozenLake-v0")
s = env.reset()
# env.render()
def epsilon_greedy(Q, s, na):
    epsilon = 0.3
    p = np.random.uniform(low = 0, high = 1)
    if p > epsilon:
        return np.argmax(Q[s,:])
    else:
        return env.action_space.sample()

Q = np.zeros([env.observation_space.n, env.action_space.n])
# Q.shape

lr = 0.5
y = 0.9
eps = 100000


for i in range(eps):

    s = env.reset()
    t = False
    while(True):
        a = epsilon_greedy(Q, s, env.action_space.n)
        s_, r, t, _ = env.step(a)
        if r == 0:
            if t == True:
                r = -5
                Q[s_] = np.ones(env.action_space.n) * r
            else:
                r = -1
        if r == 1:
            r = 100
            Q[s_] = np.ones(env.action_space.n) * r
        Q[s, a] = Q[s, a] + lr * (r + y*np.max(Q[s_, a]) - Q[s, a])
        s = s_
        if (t == True):
            break

    # if i % 10000 == 0:
    #     print(Q)

s = env.reset()
env.render()
# s
#
# Q[0]

while True:
    a = np.argmax(Q[s])
    s_, r, t, _ = env.step(a)
    print("=============")
    env.render()
    s = s_
    if t == True:
        break
