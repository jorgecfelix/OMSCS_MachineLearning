import mlrose_hiive as mlrose
import numpy as np

# Define a fitness function object.
# Define an optimization problem object.
# Select and run a randomized optimization algorithm.

def get_fitness_and_state():
    length = 1000
    # initialize fitness function
    fitness_custom = mlrose.FlipFlop()
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_custom, maximize=True, max_val=2)
    
    
    # define initial state
    init_state = np.ones(length, dtype=np.int8)
    
    return problem, init_state


def random_hill_climbing():
    """ Run random hill climbing for FlipFlop problem. """

    problem, init_state = get_fitness_and_state()

    # select random optimization problem
    # solve the problem
    best_state, best_fitness, _ = mlrose.random_hill_climb(problem,restarts=100,
                                                          max_attempts=10, max_iters=1000,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    pass


def simulated_annealing():
    """ Run simulated annealing for k-color problem. """

    problem, init_state = get_fitness_and_state()

    # define decay schedule
    schedule = mlrose.ExpDecay()
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=10, max_iters=1000,
                                                          init_state=init_state, random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


def genetic_algorithm():
    """ Genetic Algorithm for Queens problem. """
    
    problem, init_state = get_fitness_and_state()
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.1,
                                                     max_attempts=10, max_iters=1000,
                                                     random_state=1)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")


def mimic():
    """ Mimic for Queens problem. """
    
    problem, init_state = get_fitness_and_state()

    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, _ = mlrose.mimic(problem,pop_size=200, keep_pct=0.2,
                                                     max_attempts=10, max_iters=1000,
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