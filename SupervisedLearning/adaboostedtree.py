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


def ada_boosted_tree(X, y, ccp_alpha=0.0, num_samples=None, n_estimators=5):
    """ Function using scikit-learn's AdaBoostClassifier """
    print("\n :: Ada Boosted Tree")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    boosted_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME",
                         n_estimators=n_estimators)

    boosted_tree.fit(X_train, y_train)

    test_accuracy = boosted_tree.score(X_test, y_test)
    train_accuracy = boosted_tree.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")
    return train_accuracy, test_accuracy


def gradient_boosted_tree(X, y, ccp_alpha=0.0, num_samples=100):
    print("\n :: Gradient Boosted Tree")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # slice training data
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]

    boosted_tree = GradientBoostingClassifier(n_estimators=5, learning_rate=1.0,
                                              max_depth=2, random_state=0,
                                              ccp_alpha=ccp_alpha)
    boosted_tree.fit(X_train, y_train)

    test_accuracy = boosted_tree.score(X_test, y_test)
    train_accuracy = boosted_tree.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")

    return test_accuracy


def get_validation_curve(file_name, dataset_to_use):

    if dataset_to_use == 'd1':
        print("\n Using Dataset 1, Phishing data classification...")
        X, y = helper.format_phishing_data(file_name)
        train_samples = np.arange(100, 3050, 50)
    elif dataset_to_use == 'd2':
        print("\n Using Dataset 2, Bank loan data classification...")
        X, y = helper.format_bank_data(file_name)
        train_samples = np.arange(100, 2712, 50)
    else:
        print("not a valid dataset number please use d1 or d2")

    test_accuracy_data = []
    train_accuracy_data = []
    num_estimators = [i for i in range(1,100)]

    # run ada boosted on different num_estimators
    for n_estimator in num_estimators:
        print(f'\n Number of estimators used => {n_estimator}')
        train_accuracy, test_accuracy = ada_boosted_tree(X, y, n_estimators=n_estimator)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get validation curve
    plt.figure(0)
    plt.plot(num_estimators, train_accuracy_data, "-", label="train")
    plt.plot(num_estimators, test_accuracy_data, "-", label="test")
    plt.xlabel("number of estimators")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : number of estimators vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_adaboostedtree_validation.png")


def get_learning_curve(file_name, dataset_to_use):


    if dataset_to_use == 'd1':
        print("\n Using Dataset 1, Phishing data classification...")
        X, y = helper.format_phishing_data(file_name)
        train_samples = np.arange(100, 3050, 50)
    elif dataset_to_use == 'd2':
        print("\n Using Dataset 2, Bank loan data classification...")
        X, y = helper.format_bank_data(file_name)
        train_samples = np.arange(100, 2712, 50)
    else:
        print("not a valid dataset number please use d1 or d2")

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = ada_boosted_tree(X, y, num_samples=num_samples, n_estimators=20)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure(1)
    plt.plot(train_samples, train_accuracy_data, "-", label="train")
    plt.plot(train_samples, test_accuracy_data, "-", label="test")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : number training samples vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_adaboostedtree_learning.png")

if __name__ == "__main__":

    file_name = sys.argv[2]
    dataset_to_use = sys.argv[1]
    get_validation_curve(file_name, dataset_to_use)

    get_learning_curve(file_name, dataset_to_use)