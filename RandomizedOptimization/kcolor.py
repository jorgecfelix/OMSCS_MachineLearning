import mlrose_hiive as mlrose
import numpy as np

# Define a fitness function object.
# Define an optimization problem object.
# Select and run a randomized optimization algorithm.
 
# NOTE: This code is based on the tutorial from mlrose, modifed to fit a specific problem
# NOTE: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html


def get_fitness_and_state():
    num_colors = 2
    num_nodes = 5
    # initialize fitness function
    edges = [(0,1), (0,2), (0,4), (1,3), (2,3), (2,4), (3,4)]
    
    fitness = mlrose.MaxKColor(edges)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=num_nodes, fitness_fn=fitness, maximize=False, max_val=num_colors)
    
    # select random optimization problem
    
    
    # define initial state
    init_state = np.array([1, 1, 1, 1, 1])

    return problem, init_state

def random_hill_climbing():
    """ Run random hill climbing for k-color problem. """
    problem, init_state = get_fitness_and_state()

    # solve the problem
    best_state, best_fitness, _ = mlrose.random_hill_climb(problem,restarts=10,
                                                          max_attempts=100, max_iters=100,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


def simulated_annealing():
    """ Run simulated annealing for k-color problem. """

    problem, init_state = get_fitness_and_state()

    # define decay schedule
    schedule = mlrose.ExpDecay()
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=100, max_iters=1000,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


def genetic_algorithm():
    """ Genetic Algorithm for Queens problem. """
    
    problem, init_state = get_fitness_and_state()
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.1,
                                                     max_attempts=100, max_iters=1000,
                                                     random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


def mimic():
    """ Mimic for Queens problem. """
    
    problem, init_state = get_fitness_and_state()

    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.mimic(problem,pop_size=200, keep_pct=0.2,
                                                     max_attempts=100, max_iters=1000,
                                                     random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


if __name__ == "__main__":

    print("\n:: KColor Problem: ")
    print("\n:: Random Hill Climbing... ")
    random_hill_climbing()

    print("\n:: Simulated Annealing... ")
    simulated_annealing()

    print("\n:: Genetic Algorithm... ")
    genetic_algorithm()

    print("\n:: Mimic... ")
    mimic()