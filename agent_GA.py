"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt
from EnvmtDQN import environment
import time

environment = environment()
DNA_SIZE = 10          # DNA length
POP_SIZE = 100         # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 2000
X_BOUND = [0, (environment.match_num-1)]         # x upper and lower bounds
Y_BOUND = [0, environment.num_c]


def F(i,j):
    value = 0.6 * (environment.norm_match_q[i, j] + environment.norm_match_p[i, j] + environment.norm_match_f[i, j]) / \
             0.4 * (environment.norm_match_et[i, j] + environment.norm_match_wt[i, j] + environment.norm_match_ltt[i, j] + environment.norm_match_rp[i, j] + environment.norm_match_lpt[i, j])

    return value     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

# plt.ion()       # something about plotting
# x = np.linspace(*X_BOUND, environment.match_num).astype(int)
# plt.plot(x, F(x))
t_start = time.time()
quality_events = np.zeros((environment.num_c, N_GENERATIONS))
for i in range(environment.num_c):
    # print(" %d  Episode:" % i)
    plt.ion()  # something about plotting
    x = np.linspace(*X_BOUND, environment.match_num).astype(int)
    # plt.plot(x, F(i, x))
    for _ in range(N_GENERATIONS):
        y = translateDNA(pop).astype(int)
        F_values = F(i, y)    # compute function value by extracting DNA

        # something about plotting
        # if 'sca' in globals(): sca.remove()
        # sca = plt.scatter(translateDNA(pop).astype(int), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

        # GA part (evolution)
        fitness = get_fitness(F_values)
        # finess
        # for i in range(POP_SIZE):
        #     if fitness[i] <= 0:
        #         fitness[i] = 0.000000001
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child       # parent is replaced by its child
        quality_events[i][_] = sum(F_values) / len(F_values)
        # if (_ >= 0) and (_ <= 100):
        #     quality_events[i][_] = sum(F_values)/len(F_values)
        # if (_ > 100) and (_ <= 150):
        #     quality_events[i][_] = sum(F_values)/len(F_values)
        # if (_ > 150) and (_ < 200):
        #     quality_events[i][_] = max(F_values)
t_end = time.time()
print('End Time:', time.localtime(t_end))
run_time = (t_end - t_start)
print('Long-Time:', run_time)
quality_value = []
for i in range(N_GENERATIONS):
    quality_value.append(np.sum(quality_events[0: environment.num_c - 1, i]))
quality_value_arr = np.array(quality_value)

plt.figure(1)
plt.plot(quality_value)
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.grid(axis="y",linestyle = '--')
plt.show()



