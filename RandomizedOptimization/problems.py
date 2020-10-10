import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt


def ones_max(problem_size=10):
    length = problem_size

    fitness = mlrose.OneMax()

    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

    # define initial state
    init_state = np.zeros(length, dtype=np.int8)
    
    return problem, init_state


def flip_flop(problem_size=10):
    length = problem_size
    # initialize fitness function
    fitness = mlrose.FlipFlop()
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    
    
    # define initial state
    init_state = np.ones(length, dtype=np.int8)
    
    return problem, init_state


def four_peaks(problem_size=10):
    length = problem_size
    # initialize fitness function
    fitness = mlrose.FourPeaks(t_pct=0.10)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    
    np.random.seed(1)
    # define initial state
    init_state = np.random.randint(2, size=length, dtype=np.int8)
    
    return problem, init_state

def k_color(problem_size=10):
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

def queens(problem_size=10):
       # initialize fitness function
    fitness_custom = mlrose.CustomFitness(queens_max)
    
    # define optimization problem
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_custom, maximize=True, max_val=8)
    
    # define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    return problem, init_state
