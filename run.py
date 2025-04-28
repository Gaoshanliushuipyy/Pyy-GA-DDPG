# import gym
from Envmtddpg import environment
# from environment import environment
import matplotlib.pyplot as plt
# from DDPGMF import DDPG
from ddpg1 import DDPG
import time
import numpy as np

# GAME = 'Pendulum-v0'
MAX_STEP = 2
num_episodes = 2000 # 5000 0= 3000
learn_interval = 500
global_step = 0
memory_size = 3000
environment = environment()
# env = env.unwrapped
# env.seed(1)

# n_states = env.observation_space.shape[0]
# n_actions = env.action_space.shape[0]
# action_low_bound = env.action_space.low
# action_high_bound = env.action_space.high

n_states = environment.observation
n_actions = environment.action
action_low_bound = 0
action_high_bound = environment.match_num - 1

agent = DDPG(n_states, n_actions, action_low_bound, action_high_bound)

t_start = time.time()
# plt.ion()
# total_r = 0
# avg_ep_r_hist = []
a4 = []
pq1, pq2, pq3, pq4, pq6 = [], [], [], [], []
wq = []
wq1, wq2, wq3, wq4, wq5, wq6 = [], [], [], [], [], []
reward_DDPG = 1500
Tq, Ts, Tf = [30],[30],[30]
Tt, Tc = [30],[30]
for episode in range(num_episodes):
    print(" %d  Episode:" % episode)
    ep_step = 0
    # ep_r = 0
    a4.append(reward_DDPG)
    pq1.append(Tq)
    pq2.append(Ts)
    pq3.append(Tf)
    wq.append(Tt)
    wq1.append(Tc)
    environment.reset()
    observation = environment.getState()
    # else:
    #     observation = observation_
    # ep_r = 0
    global_step = 0
    # job_c = 1
    while True:
        global_step += 1
        # RL choose action based on observation
        # done, job_attrs = environment.workload(job_c)
        # observation = environment.getState(job_attrs)
        action_n = agent.choose_action(observation)
        # a = agent.choose_action(s)
        # print("a", a)
        # s_, r, done, info = env.step(a)
        # agent.store_memory(s, a, r, s_)
        observation_, reward_DDPG = environment.feedback(action_n, episode)
        agent.store_memory(observation, action_n, reward_DDPG, observation_)
        # ep_r += r
        # total_r += r
        ep_step += 1

        if agent.memory_counter >= agent.batch_size:
        # # if agent.memory_counter >= 1000:
            agent.learn()


        # if global_step != 1:
        #     agent.store_memory(observation, action_n, reward_DDPG, observation_)
            # if (global_step > start_learn) and (global_step % learn_interval == 0):
            #     brainRL.learn()
        # if (global_step > memory_size) and (global_step % learn_interval == 0):
        #     agent.learn()

        observation = observation_
        Tq, Ts, Tf = environment.feedback1()
        Tt, Tc = environment.feedback2()
        # ep_r += reward_DDPG
        if global_step >= MAX_STEP:
            break

t_end = time.time()
print('End Time:', time.localtime(t_end))
run_time = (t_end - t_start)       # 时间单位为秒
print('Long-Time:', run_time)


x_r = np.zeros(num_episodes, dtype=float)
y_r = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_r[i] = i
#将a4弄成横轴的数组放到y_r中
a4_arr = np.array(a4)

for i in range(num_episodes):
    y_r[i] = a4_arr[i]
# 每performance_showT步求平均放入y_r1中
y_r1 = []
performance_c_time = 1
for i in range(num_episodes):
    performance_showT = 1
    if i == performance_c_time * performance_showT:
        y_r1.append(np.mean(y_r[(performance_c_time - 1) * performance_showT: performance_c_time * performance_showT]))
        performance_c_time += 1

plt.figure(1)
plt.plot(y_r1)
plt.xlabel('Episodes', fontsize = 13)
plt.ylabel('Cumulative reward', fontsize = 13)
plt.grid(axis="y",linestyle = '--')
#坐标轴粗细
ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1)#设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)#设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)#设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)#设置上部坐标轴的粗细

plt.show()

'''总的时间'''
x_t = np.zeros(num_episodes, dtype=float)
y_t = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_t[i] = i
#将a4弄成横轴的数组放到y_r中
wq_arr = np.array(wq)
for i in range(num_episodes):
    y_t[i] = wq_arr[i]
# 每performance_showT步求平均放入y_r1中
plt.figure(1)
plt.plot(y_t)
plt.xlabel('episodes')
plt.ylabel('TT')
plt.grid(axis="y",linestyle = '--')
plt.show()

'''总的花费'''
x_c = np.zeros(num_episodes, dtype=float)
y_c = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_c[i] = i
#将a4弄成横轴的数组放到y_r中
wq1_arr = np.array(wq1)

for i in range(num_episodes):
    y_c[i] = wq1_arr[i]
# 每performance_showT步求平均放入y_r1中
plt.figure(1)
plt.plot(y_c)
plt.xlabel('episodes')
plt.ylabel('TC')
plt.grid(axis="y",linestyle = '--')
plt.show()

'''总的质量'''
x_q = np.zeros(num_episodes, dtype=float)
y_q = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_q[i] = i
#将a4弄成横轴的数组放到y_r中
pq1_arr = np.array(pq1)

for i in range(num_episodes):
    y_q[i] = pq1_arr[i]
# 每performance_showT步求平均放入y_r1中
plt.figure(1)
plt.plot(y_q)
plt.xlabel('episodes')
plt.ylabel('TQ')
plt.grid(axis="y",linestyle = '--')
plt.show()

'''总的满意度'''
x_s = np.zeros(num_episodes, dtype=float)
y_s = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_s[i] = i
#将a4弄成横轴的数组放到y_r中
pq2_arr = np.array(pq2)

for i in range(num_episodes):
    y_s[i] = pq2_arr[i]
#
plt.figure(1)
plt.plot(y_s)
plt.xlabel('episodes')
plt.ylabel('TS')
plt.grid(axis="y",linestyle = '--')
plt.show()

'''总的环境友好'''
x_f = np.zeros(num_episodes, dtype=float)
y_f = np.zeros(num_episodes, dtype=float)
for i in range(num_episodes):
    x_f[i] = i

pq3_arr = np.array(pq3)

for i in range(num_episodes):
    y_f[i] = pq3_arr[i]

plt.figure(1)
plt.plot(y_f)
plt.xlabel('episodes')
plt.ylabel('TF')
plt.grid(axis="y",linestyle = '--')
plt.show()