Code can be found in a public Github repository link below.

https://github.com/jorgecfelix/OMSCS_MachineLearning

In the repository, the MarkovDecisionProcesses directory contains all the necessary code
for this fourth assignment.

For this assignment I wrote indvidual python scripts to run each of the experiments and create the plots seperated by MDP:

    - frozenlake.py
    - forestmanagement.py

Each of the above scripts creates and saves named plots as pngs for each part of the assignment, description below

    frozenlake.py:
        This script runs Policy Iteration, Value Iteration, and QLearning on the Frozen Lake MDP.

    forestmanagement.py
        This script runs Policy Iteration, Value Iteration, and QLearning on the Forest Management MDP.

Python 3.7.0 was used for this assignment with the packages and versions below:
    - pandas==0.24.2
    - numpy==1.16.0
    - scikit-learn==0.23.2
    - matplotlib==3.3.1
    - gym==0.17.3
    - mdptoolbox-hiive==4.0.3.1



    
MDPS needed:

The Frozen Lake MDP is described and can be found here: https://gym.openai.com/envs/#toy_text
The code takes care of importing and creating the P and R matrices needed for mdptoolbox.

The Forest Management MDP is a part of mdptoolbox and the code handles the creation of the P and R matrices.

How to run:

Each script takes in at least one argument

The first argument must be either pi, vi, or ql which runs either Policy Iteration, Value Iteration, and Q-Learning.

If running with the ql argument then there must be a second argument that has to be gamma, alpha, or epsilon which 
tunes either of those parameters and outputs the respective plots.


The scripts will then run  and output plots as png files.


Examples for running a script with Policy or Value Iteration:

    python forestmanagement.py pi

    python frozenlake.py vi

Examples for running a script with Q-Learning

    python forestmanagement.py ql gamma

    python forestmanagement.py ql alpha
    
    python frozenlake.py ql epsilon
