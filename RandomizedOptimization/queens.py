import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt

# Define a fitness function object.
# Define an optimization problem object.
# Select and run a randomized optimization algorithm.
# NOTE: This code does not belong or was written by me. 
# NOTE: This code is based on the tutorial from mlrose
# NOTE: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html


# fitness function
def queens_max(state):
    fitness_cnt = 0

    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):  
            # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):

                   # If no attacks, then increment counter
                   fitness_cnt += 1
    return fitness_cnt

def get_fitness_and_state():
       # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    return problem, init_state

def random_hill_climbing(max_attempts=100, max_iters=100):
    """ Run random hill climbing for FourPeaks problem. """

    problem, init_state = get_fitness_and_state()

    # select random optimization problem
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts=0,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    
    return fitness_curve


def simulated_annealing(max_attempts=100, max_iters=100):
    """ Run simulated annealing for k-color problem. """

    problem, init_state = get_fitness_and_state()

    # define decay schedule
    schedule = mlrose.ExpDecay()
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

    return fitness_curve


def genetic_algorithm(max_attempts=100, max_iters=100):
    """ Genetic Algorithm for Queens problem. """
    
    problem, init_state = get_fitness_and_state()
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.1,
                                                     max_attempts=max_attempts, max_iters=max_iters,
                                                     random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

    return fitness_curve

def mimic(max_attempts=100, max_iters=100):
    """ Mimic for Queens problem. """
    
    problem, init_state = get_fitness_and_state()

    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem,pop_size=200, keep_pct=0.2,
                                                     max_attempts=max_attempts, max_iters=max_iters,
                                                     random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

    return fitness_curve

if __name__ == "__main__":

    print("\n:: FourPeaks Problem: ")

    max_attempts=100
    max_iters=100

    print("\n:: Random Hill Climbing... ")
    fc_1 = random_hill_climbing(max_attempts=max_attempts, max_iters=max_iters)

    print("\n:: Simulated Annealing... ")
    fc_2 = simulated_annealing(max_attempts=max_attempts, max_iters=max_iters)

    print("\n:: Genetic Algorithm... ")
    fc_3 = genetic_algorithm(max_attempts=max_attempts, max_iters=max_iters)

    print("\n:: Mimic... ")
    fc_4 = mimic(max_attempts=max_attempts, max_iters=max_iters)

    plt.plot(fc_1, label='rand_hc')
    plt.plot(fc_2, label='sim_a')
    plt.plot(fc_3, label='gen_alg')
    plt.plot(fc_4, label='mimi')
    plt.legend(loc="lower right")
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    plt.show()