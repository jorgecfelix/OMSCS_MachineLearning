import mlrose_hiive as mlrose
import numpy as np

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


def simulated_annealing():
    """ Run simulated annealing for Queens problem. """
    
    # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # select random optimization problem
    
    # define decay schedule
    schedule = mlrose.ExpDecay()
    
    # define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=100, max_iters=1000,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

def random_hill_climbing():
    """ Run random hill climbing for Queens problem. """
    
    # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # select random optimization problem
    
    
    # define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.random_hill_climb(problem,restarts=100,
                                                          max_attempts=100, max_iters=1000,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

def genetic_algorithm():
    """ Genetic Algorithm for Queens problem. """
    
    # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.1,
                                                     max_attempts=100, max_iters=1000,
                                                     random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

def mimic():
    """ Mimic for Queens problem. """
    
    # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.mimic(problem,pop_size=200, keep_pct=0.2,
                                                     max_attempts=100, max_iters=1000,
                                                     random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")

if __name__ == "__main__":

    print("\n:: Queens Problem: ")
    print("\n:: Random Hill Climbing... ")
    random_hill_climbing()

    print("\n:: Simulated Annealing... ")
    simulated_annealing()

    print("\n:: Genetic Algorithm... ")
    genetic_algorithm()

    print("\n:: Mimic... ")
    mimic()