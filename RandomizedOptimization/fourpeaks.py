import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
from problems import four_peaks


def random_hill_climbing(problem, problem_size=10, max_attempts=100, max_iters=100):
    """ Run random hill climbing """

    problem, init_state = problem(problem_size=problem_size)

    # select random optimization problem
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts=100,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    # print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)


def simulated_annealing(problem, problem_size=10, max_attempts=100, max_iters=100, decay=0.55):
    """ Run simulated annealing """

    problem, init_state = problem(problem_size=problem_size)

    # define decay schedule
    schedule = mlrose.ExpDecay(init_temp=100, min_temp=0.6, exp_const=decay)
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=max_attempts, max_iters=max_iters,
                                                          init_state=init_state, random_state=1, curve=True)
    # print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)


def genetic_algorithm(problem, problem_size=10, max_attempts=100, max_iters=100):
    """ Genetic Algorithm. """
    
    problem, init_state = problem(problem_size=problem_size)
    # select random optimization problem
    
    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,pop_size=100, mutation_prob=0.1,
                                                     max_attempts=max_attempts, max_iters=max_iters, 
                                                     random_state=1, curve=True)
    # print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)

def mimic(problem, problem_size=10, max_attempts=100, max_iters=100):
    """ Mimic. """
    
    problem, init_state = problem(problem_size=problem_size)

    # select random optimization problem

    # solve the problem
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=100, keep_pct=0.1,
                                                     max_attempts=max_attempts, max_iters=max_iters,
                                                     random_state=1, curve=True)
    # print(f":: Best State: {best_state}")
    print(f":: Best Fitness: {best_fitness}")
    print(f":: Number of iterations {len(fitness_curve)}")

    return fitness_curve, best_fitness, len(fitness_curve)

def get_fitness_vs_iterations(problem_size):

    max_attempts=10
    max_iters=100
    problems = [four_peaks]
    names = ['four_peaks']

    for index, problem in enumerate(problems):
        print(f"\n\n:: {names[index]}... ")
        fc_rhc, _, _ = random_hill_climbing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
    
        print("\n:: Simulated Annealing... ")
        fc_sa, _, _ = simulated_annealing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
    
        print("\n:: Genetic Algorithm... ")
        fc_ga, _, _ = genetic_algorithm(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
    
        print("\n:: Mimic... ")
        fc_m, _, _ = mimic(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
    
        # create new figure
        plt.figure()
    
        plt.plot(fc_rhc, label='rand_hc')
        plt.plot(fc_sa, label='sim_a')
        plt.plot(fc_ga, label='gen_alg')
        plt.plot(fc_m, label='mimic')
        plt.legend(loc="lower right")
        plt.xlabel("iterations")
        plt.ylabel("fitness")
        plt.savefig(f"iters_vs_fitness_{names[index]}.png")


def get_fitness_vs_problem_size(sizes):

    problems = [four_peaks]
    names = ['four_peaks']
    max_attempts = 100
    max_iters = np.inf
 
    for index, problem in enumerate(problems):
        print(f"\n\n:: {names[index]}... ")
        # keep track of fitness for each size
        fitness_rhc = []
        fitness_sa = []
        fitness_ga = []
        fitness_mim = []

        # loop through different sizes
        for size in sizes:
            
            problem_size = size
            print(f"\n\n:: Problem Size {size}... ")
            print("\n:: Random Hill Climbing...")
            _, fit_rhc, _ = random_hill_climbing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Simulated Annealing... ")
            _, fit_sa, _ = simulated_annealing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Genetic Algorithm... ")
            _, fit_ga, _ = genetic_algorithm(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Mimic... ")
            _, fit_mim, _ = mimic(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
            
            fitness_rhc.append(fit_rhc)
            fitness_sa.append(fit_sa)
            fitness_ga.append(fit_ga)
            fitness_mim.append(fit_mim)

        # create new figure
        plt.figure()
    
        plt.plot(sizes, fitness_rhc, label='rand_hc')
        plt.plot(sizes, fitness_sa, label='sim_a')
        plt.plot(sizes, fitness_ga, label='gen_alg')
        plt.plot(sizes, fitness_mim, label='mimic')
        plt.legend(loc="lower right")
        plt.xlabel("problem_size")
        plt.ylabel("fitness")
        plt.savefig(f"problemsize_vs_fitness_{names[index]}.png")

def get_problem_size_vs_evals(sizes):

    problems = [four_peaks]
    names = ['four_peaks']
    max_attempts = 100
    max_iters = np.inf
 
    for index, problem in enumerate(problems):
        print(f"\n\n:: {names[index]}... ")
        # keep track of fitness for each size
        evals_rhc = []
        evals_sa = []
        evals_ga = []
        evals_mim = []

        # loop through different sizes
        for size in sizes:
            
            problem_size = size
            print(f"\n\n:: Problem Size {size}... ")
            print("\n:: Random Hill Climbing...")
            _, fit_rhc, eval_rhc = random_hill_climbing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Simulated Annealing... ")
            _, fit_sa, eval_sa = simulated_annealing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Genetic Algorithm... ")
            _, fit_ga, eval_ga = genetic_algorithm(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
        
            print("\n:: Mimic... ")
            _, fit_mim, eval_mim = mimic(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters)
            
            evals_rhc.append(eval_rhc)
            evals_sa.append(eval_sa)
            evals_ga.append(eval_ga)
            evals_mim.append(eval_mim)

        # create new figure
        plt.figure()
    
        plt.plot(sizes, evals_rhc, label='rand_hc')
        plt.plot(sizes, evals_sa, label='sim_a')
        plt.plot(sizes, evals_ga, label='gen_alg')
        plt.plot(sizes, evals_mim, label='mimic')
        plt.legend(loc="lower right")
        plt.xlabel("problem_size")
        plt.ylabel("evaluations")
        plt.savefig(f"problemsize_vs_evals_{names[index]}.png")
    pass


def tune_simulated_annealing():
    print("\n:: Simulated Annealing Decay Tuning... ")
    decays = np.arange(0.01, 1.0, 0.05)
    problems = [four_peaks]
    names = ['four_peaks']
    max_attempts = 100
    max_iters = np.inf
    problem_size = 100

    for index, problem in enumerate(problems):
        fitness_decay = []
    
        for decay in decays:
            fitness_curve, fit_sa, eval_sa = simulated_annealing(problem, problem_size=problem_size, max_attempts=max_attempts, max_iters=max_iters, decay=decay)
            fitness_decay.append(fit_sa)
        plt.figure()

        plt.plot(decays, fitness_decay, label='sim_a')
        plt.legend(loc="lower right")
        plt.xlabel("decay")
        plt.ylabel("fitness")
        plt.savefig(f"decay_vs_fitness_{names[index]}.png")


if __name__ == "__main__":
    np.random.seed(10)

    sizes = range(10, 110, 10)
    problem_size = 100
    get_fitness_vs_iterations(problem_size)
    #get_fitness_vs_problem_size(sizes)
    #get_problem_size_vs_evals(sizes)

    tune_simulated_annealing()