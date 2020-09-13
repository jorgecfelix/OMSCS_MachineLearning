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
from helper import read_csv_data, split_data, format_phishing_data


def run_decision_tree(X, y, test_size=0.60, ccp_alpha=0.0):

    """ Function used to fit decision tree with a certain test size and ccp_alpha. """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create decision tree classifier and fit to training data
    decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha,random_state=0)
    decision_tree.fit(X_train, y_train)

    test_accuracy = decision_tree.score(X_test, y_test)
    train_accuracy = decision_tree.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")

    return X_train.shape[0], test_accuracy, train_accuracy


def complex_post_prune_tree(X, y):
    """ Complex Post Pruning on decision tree,
        example retrieved from scikit-learn official documentation

        Get the alpha values and its impurites at each alpha,
        then train decision tree at each alpha.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

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

    # plot accuracy of test and train vs alpha used in decision tree to train
    plt.plot(ccp_alphas[:-1], train_scores[:-1], "-o", label="train")
    plt.plot(ccp_alphas[:-1], test_scores[:-1], "-o", label="test")
    plt.xlabel("effective alpha")
    plt.ylabel("Accuracy ")
    plt.legend(loc="upper left")
    plt.show()
    best_ccp_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    print(f"Max Alpha is {best_ccp_alpha}")

    return trees, best_ccp_alpha


def plot_decision_tree(decision_tree):
    _, ax = plt.subplots(figsize=(10, 10))  # whatever size you want
    tree.plot_tree(decision_tree, ax=ax)
    plt.show()


def ada_boosted_tree(X, y):
    """ Function using scikit-learn's AdaBoostClassifier """
    print("\n :: Ada Boosted Tree")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    boosted_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME",
                         n_estimators=1)

    boosted_tree.fit(X_train, y_train)

    test_accuracy = boosted_tree.score(X_test, y_test)
    train_accuracy = boosted_tree.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")
    return boosted_tree


def gradient_boosted_tree(X, y):
    print("\n :: Gradient Boosted Tree")
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    boosted_tree = GradientBoostingClassifier(n_estimators=2, learning_rate=1.0, max_depth=2, random_state=0)
    boosted_tree.fit(X_train, y_train)

    test_accuracy = boosted_tree.score(X_test, y_test)
    train_accuracy = boosted_tree.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")

    return boosted_tree
    


if __name__ == "__main__":

    # dataset = read_csv_data(sys.argv[1], delimiter=",", encode=False)
    # X, y = split_data(dataset, class_attr="class")

    # get phising data
    X, y = format_phishing_data(sys.argv[1])
    test_sizes = np.arange(.95, .05, -.05)
    
    num_samples_list = []
    accuracy_data = []
    trees, ccp_alpha = complex_post_prune_tree(X, y)

    gradient_boosted_tree(X, y)
    ada_boosted_tree(X, y)

    for test_size in test_sizes:
        num_samples, test_accuracy, train_accuracy = run_decision_tree(X, y, test_size=test_size, ccp_alpha=ccp_alpha)
        num_samples_list.append(num_samples)
        accuracy_data.append(test_accuracy)

    #sl.plot_accuracy_vs_training_samples(num_samples_list, accuracy_data)