import sys
import gym
import numpy as np
import hiive.mdptoolbox.mdp, hiive.mdptoolbox.example
import matplotlib.pyplot as plt
from pprint import pprint

def get_stats_list(stats):

    errors = [stat['Error'] for stat in stats]
    reward = [stat['Reward'] for stat in stats]
    mean_v = [stat['Mean V'] for stat in stats]
    times = [stat['Time'] for stat in stats]

    return errors, reward, mean_v, times


def run_forest_management(states=10, max_iter=1000, alg='pi', gamma=0.9):
    print(f"\n Running forest management mdp example with states={states}")
    P, R = hiive.mdptoolbox.example.forest(S=states)


    # print(P)
    if alg == 'pi':
        print(" starting policy iteration...")
        policy_iter = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=max_iter)
        stats = policy_iter.run()
        policy = policy_iter.policy
        V = policy_iter.V
    elif alg == 'vi':
        print(" starting value iteration...")
        val_iter = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=max_iter)
        stats = val_iter.run()
        policy = val_iter.policy

        V = val_iter.V
    # print stats
    #pprint(stats)

    errors, reward, mean_v, times = get_stats_list(stats)

    plt.figure()
    plt.plot(times, "-", label='time')
    plt.xlabel("iteration")
    plt.ylabel("time")
    plt.title(f" iteration vs time states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"forestM_{alg}_iter_num_vs_time_s={states}.png")

    plt.figure()
    plt.plot(errors, "-", label='error')
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title(f" iteration vs error states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"forestM_{alg}_iter_num_vs_error_s={states}.png")
    plt.close()

    plt.figure()
    plt.plot(reward, "-", label='reward')
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.title(f" iteration vs reward states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"forestM_{alg}_iter_num_vs_reward_s={states}.png")
    plt.close()

    plt.figure()
    plt.plot(mean_v, "-", label='meanV')
    plt.xlabel("iteration")
    plt.ylabel("mean V")
    plt.title(f" iteration vs Mean V states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"forestM_{alg}_iter_num_vs_meanV_s={states}.png")
    plt.close()


    plt.figure()
    plt.plot(policy, "-", label='policy')
    plt.xlabel("state")
    plt.ylabel("action")
    plt.title(f" The Policy Actions at a given State states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"forestM_{alg}_state_vs_action_s={states}.png")
    plt.close()

    return stats

def run_forest_management_gamma(states=10, max_iter=1000, alg='pi', gamma=0.9):
    print("\n Running forest management mdp example with change in gamma")
    P, R = hiive.mdptoolbox.example.forest(S=states)


    # print(P)
    if alg == 'pi':
        print(" starting policy iteration...")
        policy_iter = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=max_iter)
        stats = policy_iter.run()

    elif alg == 'vi':
        print(" starting value iteration...")
        val_iter = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=max_iter)
        stats = val_iter.run()

    # print stats
    #pprint(stats)

    errors, reward, mean_v, times = get_stats_list(stats)
    plt.figure(10)
    plt.plot(times, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("time")
    plt.title(f" iteration vs time states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"gamma_forestM_{alg}_iter_num_vs_time_s={states}.png")

    plt.figure(11)
    plt.plot(errors, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title(f" iteration vs error states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"gamma_forestM_{alg}_iter_num_vs_error_s={states}.png")

    plt.figure(12)
    plt.plot(reward, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.title(f" iteration vs reward states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"gamma_forestM_{alg}_iter_num_vs_reward_s={states}.png")

    plt.figure(13)
    plt.plot(mean_v, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("mean V")
    plt.title(f" iteration vs Mean V states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"gamma_forestM_{alg}_iter_num_vs_meanV_s={states}.png")

    return stats

def run_forest_management_ql(states=1000, value=0.99, gamma=0.99, epsilon=0.5, epsilon_decay=0.6, 
                             alpha_decay=0.99, alpha=0.6, p='gamma' ):
    print("\n Running forest management mdp example with change in gamma")
    P, R = hiive.mdptoolbox.example.forest(S=states)


    print(" starting QLearning")
    if p == 'gamma':
        qlearn = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=value, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                             alpha_decay=alpha_decay, alpha=alpha )
    elif p == 'epsilon':
        qlearn = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, epsilon=value, epsilon_decay=epsilon_decay,
                                             alpha_decay=alpha_decay, alpha=alpha )
    elif p == 'alpha':
        qlearn = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                             alpha_decay=alpha_decay, alpha=value )
    stats = qlearn.run()


    errors, reward, mean_v, times = get_stats_list(stats)
    plt.figure(10)
    plt.plot(times, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("time")
    plt.title(f" iteration vs time states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forestM_ql_iter_num_vs_time.png")

    plt.figure(11)
    plt.plot(errors, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title(f" iteration vs error states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forestM_ql_iter_num_vs_error.png")

    plt.figure(13)
    plt.plot(mean_v, "-", label=f"{gamma}")
    plt.xlabel("iteration")
    plt.ylabel("mean V")
    plt.title(f" iteration vs Mean V states={states}")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forestM_ql_iter_num_vs_meanV.png")

    return stats




if __name__ == '__main__':

    alg = sys.argv[1]
    p = sys.argv[2]

    states = 10, 100, 1000, 10000

    if alg !='ql':
        for state in states:
            run_forest_management(states=state, alg=alg)

    if alg !='ql':
        gammas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for gamma in gammas:
            run_forest_management_gamma(states=1000, alg=alg, gamma=gamma)
    else:
        gammas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for gamma in gammas:
            run_forest_management_ql(gamma=gamma, p=p)