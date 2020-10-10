import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt

# Define a fitness function object.
# Define an optimization problem object.
# Select and run a randomized optimization algorithm.

def get_fitness_and_state(problem_size=10):
    length = problem_size
    # initialize fitness function
    fitness = mlrose.FlipFlop()
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    
    
    # define initial state
    init_state = np.ones(length, dtype=np.int8)
    
    return problem, init_state


def random_hill_climbing(problem_size=10, max_attempts=100, max_iters=100):
    """ Run random hill climbing for FourPeaks problem. """

    problem, init_state = get_fitness_and_state(problem_size=problem_size)

    # select random optimization problem
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts=100,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)


def simulated_annealing(problem_size=10, max_attempts=100, max_iters=100):
    """ Run simulated annealing for k-color problem. """

    problem, init_state = get_fitness_and_state(problem_size=problem_size)

    # define decay schedule
    schedule = mlrose.GeomDecay()
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)


def genetic_algorithm(problem_size=10, max_attempts=100, max_iters=100):
    """ Genetic Algorithm for Queens problem. """
    
    problem, init_state = get_fitness_and_state(problem_size=problem_size)
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.1,
                                                     max_attempts=max_attempts, max_iters=max_iters, 
                                                     random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)

def mimic(problem_size=10, max_attempts=100, max_iters=100):
    """ Mimic for Queens problem. """
    
    problem, init_state = get_fitness_and_state(problem_size=problem_size)

    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem,pop_size=200, keep_pct=0.2,
                                                     max_attempts=max_attempts, max_iters=max_iters,
                                                     random_state=1, curve=True)
    print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)

    
def get_fitness_curve():
    print("\n:: FourPeaks Problem: ")

    max_attempts=10
    max_iters=np.inf
    problem_size = 100

    print("\n:: Random Hill Climbing... ")
    fc_rhc, _, _ = random_hill_climbing(problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)

    print("\n:: Simulated Annealing... ")
    fc_sa, _, _ = simulated_annealing(problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)

    print("\n:: Genetic Algorithm... ")
    fc_ga, _, _ = genetic_algorithm(problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)

    #print("\n:: Mimic... ")
    #fc_m, _, _ = mimic(max_attempts=max_attempts, max_iters=max_iters)

    plt.plot(fc_rhc, label='rand_hc')
    plt.plot(fc_sa, label='sim_a')
    plt.plot(fc_ga, label='gen_alg')
    #plt.plot(fc_m, label='mimic')
    plt.legend(loc="lower right")
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    plt.show()


def get_fitness_vs_problem_size():
    pass

if __name__ == "__main__":
    get_fitness_curve()