from Envmtddpg import environment
from mchgenalg import GeneticAlgorithm
# from environment import environment
import matplotlib.pyplot as plt
# from DDPGMF import DDPG
from ddpg2 import DDPG
import tensorflow as tf
import numpy as np
import os

timesEvaluated = 0
bestepochs = -1
environment = environment()
# First, define function that will be used to evaluate the fitness
data = np.zeros((330, 2000), dtype=float)
Padata = np.zeros((330, 4), dtype=float)
def fitness_function(genome):

    global timesEvaluated
    timesEvaluated += 1

    print("Fitness function invoked "+str(timesEvaluated)+" times")

    #setting parameter values using genome
    # polyak = 0.01 + 0.1 * np.tanh((decode_function(genome[0:10]) - 0.01) / 0.1)
    polyak = decode_function(genome[0:10])
    if polyak > 1:
        polyak = 1
    # gamma = 0.99 + 0.3 * np.tanh((decode_function(genome[11:21]) - 0.99) / 0.3)
    gamma = decode_function(genome[11:21])
    if gamma > 1:
        gamma = 1
    # Q_lr = 0.000005 + 0.000002 * np.tanh((decode_function(genome[22:32]) - 0.000005) / 0.000002)
    # Q_lr = 0.000001 + 0.00001 / (1 + np.exp(-1000*(decode_function_one(genome[22:32]))))# 跑好的(0.000003  0.000004)(0.000001  0.000009)
    Q_lr = decode_function_one(genome[22:32])
    if Q_lr > 1:
       Q_lr = 1
    # pi_lr = 0.000005 + 0.000002 * np.tanh((decode_function(genome[33:43]) - 0.000005) / 0.000002)
    # pi_lr = 0.000001 + 0.00001 / (1 + np.exp(-1000*(decode_function_one(genome[33:43]))))# 跑好的(0.000003  0.000004)(0.000001  0.000009)
    pi_lr = decode_function_one(genome[33:43])
    if pi_lr > 1:
        pi_lr = 1
    pa=[]
    pa.append(polyak), pa.append(gamma), pa.append(Q_lr), pa.append(pi_lr)

    MAX_STEP = 20
    num_episodes = 2000  # 5000 = 3000
    # learn_interval = 500
    # global_step = 0
    # memory_size = 3000
    # environment = environment()
    n_states = environment.observation
    n_actions = environment.action
    action_low_bound = 0
    action_high_bound = environment.match_num - 1
    tf.reset_default_graph() #这个很重要
    agent = DDPG(n_states, n_actions, action_low_bound, action_high_bound, gamma, actor_lr=Q_lr, critic_lr=pi_lr, tau=polyak) #这里要变
    a4 = []
    pq1, pq2, pq3, pq4, pq6 = [], [], [], [], []
    wq = []
    wq1, wq2, wq3, wq4, wq5, wq6 = [], [], [], [], [], []
    reward_DDPG = 1500
    Tq, Ts, Tf = [30], [30], [30]
    Tt, Tc = [30], [30]
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
    a4_arr = np.array(a4)
    # data = np.zeros((timesEvaluated, num_episodes), dtype=float)
    for i in range(num_episodes):
        data[timesEvaluated][i]=a4_arr[i]
    pa_arr = np.array(pa)
    # data = np.zeros((timesEvaluated, num_episodes), dtype=float)
    for i in range(4):
        Padata[timesEvaluated][i]=pa_arr[i]

    y_r = np.zeros(num_episodes, dtype=float)
    for i in range(num_episodes):
        y_r[i] = a4_arr[i]
    # 每performance_showT步求平均放入y_r1中
    y_r1 = []
    performance_c_time = 1
    for i in range(num_episodes):
        performance_showT = 1
        if i == performance_c_time * performance_showT:
            y_r1.append(
                np.mean(y_r[(performance_c_time - 1) * performance_showT: performance_c_time * performance_showT]))
            performance_c_time += 1

    plt.figure(1)
    plt.plot(y_r1)
    plt.xlabel('Episodes', fontsize=13)
    plt.ylabel('Cumulative reward'+str(timesEvaluated), fontsize=13)
    plt.grid(axis="y", linestyle='--')
    # 坐标轴粗细
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1)  # 设置上部坐标轴的粗细

    plt.show()
    return reward_DDPG

def decode_function(genome_partial):

    prod = 0
    for i,e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0   # False的时候prod值不变
        else:
            prod += 2**abs(i-len(genome_partial)+1)
    return prod/1000
def decode_function_one(genome_partial):

    prod = 0
    for i,e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0   # False的时候prod值不变
        else:
            prod += 2**abs(i-len(genome_partial)+1)
    return prod/10000000

# Configure the algorithm:
population_size = 30
genome_length = 43
ga = GeneticAlgorithm(fitness_function)
ga.generate_binary_population(size=population_size, genome_length=genome_length)

# How many pairs of individuals should be picked to mate
ga.number_of_pairs = 5

# Selective pressure from interval [1.0, 2.0]
# the lower value, the less will the fitness play role
ga.selective_pressure = 1.5
ga.mutation_rate = 0.1

# If two parents have the same genotype, ignore them and generate TWO random parents
# This helps preventing premature convergence
ga.allow_random_parent = True # default True
# Use single point crossover instead of uniform crossover
ga.single_point_cross_over = False # default False

# Run 100 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(1) # default 1000

best_genome, best_fitness = ga.get_best_genome()

print("BEST CHROMOSOME IS")
print(best_genome)
print("It's decoded value is")
print("Tau = " + str(decode_function(best_genome[0:10])))
print("Gamma = " + str(decode_function(best_genome[11:22])))
print("Q_learning = " + str(decode_function(best_genome[23:33])))
print("pi_learning = " + str(decode_function(best_genome[34:44])))
print("random_epsilon = " + str(decode_function(best_genome[45:55])))
print("noise_epsilon = " + str(decode_function(best_genome[56:66])))

# If you want, you can have a look at the population:
population = ga.population

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()
np.savetxt("./dataone/ImproveR1.txt", data, fmt="%f", delimiter="  ")
np.savetxt("./dataone/ImproveR2.txt", Padata, fmt="%f", delimiter="  ")

