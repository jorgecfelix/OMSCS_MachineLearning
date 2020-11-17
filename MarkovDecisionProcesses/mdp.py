import gym
import numpy as np
import hiive.mdptoolbox as mdptoolbox
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

if __name__ == '__main__':
    create_frozen_lake_P_and_R('FrozenLake-v0')
    create_frozen_lake_P_and_R('FrozenLake8x8-v0')