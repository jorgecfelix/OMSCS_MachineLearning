import sys
import gym
import numpy as np
import hiive.mdptoolbox.mdp, hiive.mdptoolbox.example
import matplotlib.pyplot as plt
from pprint import pprint
from gym.envs.toy_text.frozen_lake import generate_random_map


# print(gym.envs.registry.all())

def create_frozen_lake_P_and_R(size='FrozenLake-v0'):

    print(f"\n Creating {size} P and R matrices")
    env = gym.make(size)
    
    print(f" Num states : {env.nS}")
    print(f" Num actions : {env.nA}")
    s = env.nS
    a = env.nA

    # pprint(env.P)
    
    # construct P and R to be used in mdptoolbox
    
    # initialize P and R with zeroes
    P = np.zeros((a, s, s))
    R = np.zeros((s, a))
    
    # loop through openai gym P environment given and construct P and R for mdptoolbox
    for state in env.P:
        for action in env.P[state]:
            #print("\n\n")
            for prob, sprime, reward, done in env.P[state][action]:
                # print(prob,sprime, reward, done)
                # populate P and R matrix
                P[action][state][sprime] += prob
                R[state][action] = reward
    
    # print(P)
    # print(R)

    # check for 0 rows in P matrix to get probability

    print("\n Checking for zero probability lists in P matrix")
    for action in P:
        # print("\n")
        # go through each prob_list and check for zeros sums in probability
        for prob_list in action:
            # print("\n")
            # print(sum(prob_list))
            if sum(prob_list) == 0.0:
                print(f" Found at action {action} and list {prob_list}")
                
    # return new P and R matrices
    return P, R

def get_stats_list(stats):

    errors = [stat['Error'] for stat in stats]
    reward = [stat['Reward'] for stat in stats]
    mean_v = [stat['Mean V'] for stat in stats]
    times = [stat['Time'] for stat in stats]

    return errors, reward, mean_v, times

def run_frozen_lake(size='FrozenLake8x8-v0',max_iter=100, alg='pi', gamma=0.9):
    print("\n Running frozen lake mdp example")
    P, R = create_frozen_lake_P_and_R(size)
    
    if alg == 'pi':
        print(" starting policy iteration")
        policy_iter = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=max_iter)
        stats = policy_iter.run()
    elif alg == 'vi':
        print(" starting value iteration")
        val_iter = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=max_iter)
        stats = val_iter.run()
    
    # print stats
    errors, reward, mean_v, times = get_stats_list(stats)

    plt.figure(10)
    plt.plot(times, "-o", label=f'{gamma}')
    plt.xlabel("iteration")
    plt.ylabel("time")
    plt.title(f" iteration vs time {size}")
    plt.legend(loc="upper right")
    plt.savefig(f"{size}_{alg}_iter_num_vs_time.png")

    plt.figure(11)
    plt.plot(errors, "-o", label=f'{gamma}')
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title(f" iteration vs error {size}")
    plt.legend(loc="upper right")
    plt.savefig(f"{size}_{alg}_iter_num_vs_error.png")

    plt.figure(12)
    plt.plot(reward, "-o", label=f'{gamma}')
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.title(f" iteration vs reward {size}")
    plt.legend(loc="upper right")
    plt.savefig(f"{size}_{alg}_iter_num_vs_reward.png")

    plt.figure(13)
    plt.plot(mean_v, "-o", label=f'{gamma}')
    plt.xlabel("iteration")
    plt.ylabel("mean V")
    plt.title(f" iteration vs Mean V {size}")
    plt.legend(loc="upper right")
    plt.savefig(f"{size}_{alg}_iter_num_vs_meanV.png")


if __name__ == '__main__':
    alg = sys.argv[1]

    gammas=[0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8, 0.9]
    for gamma in gammas:
        run_frozen_lake('FrozenLake-v0', alg=alg, gamma=gamma)
