import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import helper

# global used for print statements
DEBUG = True

def decision_tree_learning(X_train, X_test, y_train, y_test, num_samples=None, ccp_alpha=0.0):

    """ Function used to fit decision tree with a certain test size and ccp_alpha. """

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    print( "\n Number of values per class attribute used to Train:")
    print(y_train.value_counts())

    # create decision tree classifier and fit to training data
    decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=0)

    # fit on specific number of training samples
    decision_tree.fit(X_train, y_train)

    test_accuracy = decision_tree.score(X_test, y_test)
    train_accuracy = decision_tree.score(X_train, y_train)

    if DEBUG:
        print(f" Number of training samples used {X_train.shape[0]}")
        print(f" Number of test samples used for validation {X_test.shape[0]}")

        print(f" Accuracy of test data (validation): {test_accuracy}")
        # print(f" Accuracy of train data : {train_accuracy}\n")
    
    # return test and train accuracy
    return train_accuracy, test_accuracy


def complex_post_prune_tree(X_train, X_test, y_train, y_test, dataset_to_use='d1'):
    """ Complex Post Pruning on decision tree,
        example retrieved from scikit-learn official documentation

        Get the alpha values and its impurites at each alpha,
        then train decision tree at each alpha.
    """
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

    print( "\n Number of values per class attribute used to Train:")
    print(y_train.value_counts())
    print( "\n Number of values per class attribute used to Test:")
    print(y_test.value_counts())

    # NOTE: The higher the alpha the more the tree is pruned
    decision_tree = DecisionTreeClassifier(random_state=0)
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, _ = path.ccp_alphas, path.impurities

    trees = []
    for ccp_alpha in ccp_alphas:
        decision_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        decision_tree.fit(X_train, y_train)
        trees.append(decision_tree)

    # get accuracy scores for train and test data
    train_scores = [tree.score(X_train, y_train) for tree in trees]
    test_scores = [tree.score(X_test, y_test) for tree in trees]   

    # get validation curve with different alphas
    # plot accuracy of test and train vs alpha used in decision tree to train
    plt.figure()
    plt.plot(ccp_alphas[:-1], train_scores[:-1], "-", label="train")
    plt.plot(ccp_alphas[:-1], test_scores[:-1], "-", label="test")
    plt.xlabel("effective alpha")
    plt.ylabel("Accuracy ")
    plt.title(f"{dataset_to_use} : ccp alpha vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_decisiontree_validation.png")

    best_ccp_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    print(f"Max Alpha is {best_ccp_alpha}")

    return trees, best_ccp_alpha


def plot_decision_tree(decision_tree):
    _, ax = plt.subplots(figsize=(10, 10))  # whatever size you want
    tree.plot_tree(decision_tree, ax=ax)
    plt.show()


if __name__ == "__main__":

    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    test_accuracy_data = []
    train_accuracy_data = []
    print(":: Decision Tree :: Running Complex Prune ")
    # complex post prune on data and get best alpha
    trees, ccp_alpha = complex_post_prune_tree(X_train, X_test, y_train, y_test, dataset_to_use=dataset_to_use)

    print(":: Decision Tree :: Running decision tree learning with best ccp alpha ... ")

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = decision_tree_learning(X_train, X_test, y_train, y_test, num_samples=num_samples, ccp_alpha=ccp_alpha)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure()
    plt.plot(train_samples_list, train_accuracy_data, "-", label="train")
    plt.plot(train_samples_list, test_accuracy_data, "-", label="test")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} number training samples vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_decisiontree_learning.png")