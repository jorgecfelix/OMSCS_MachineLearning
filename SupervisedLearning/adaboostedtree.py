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


def ada_boosted_tree(X_train, X_test, y_train, y_test, ccp_alpha=0.0, num_samples=None, n_estimators=5):
    """ Function using scikit-learn's AdaBoostClassifier """
    print("\n :: Ada Boosted Tree")

    # split data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    print( "\n Number of values per class attribute used to Train:")
    print(y_train.value_counts())
    print( "\n Number of values per class attribute used to Test:")
    print(y_test.value_counts())

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


def get_validation_curve(file_name, dataset_to_use):

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    # split training for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.40)

    test_accuracy_data = []
    train_accuracy_data = []
    num_estimators = [i for i in range(1,100)]

    # run ada boosted on different num_estimators
    for n_estimator in num_estimators:
        print(f'\n Number of estimators used => {n_estimator}')
        train_accuracy, test_accuracy = ada_boosted_tree(X_train, X_test, y_train, y_test, n_estimators=n_estimator)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get best estimator used
    best_n_estimator = num_estimators[test_accuracy_data.index(max(test_accuracy_data))]
    
    print(f"\n :: adaboost :: Best estimator to use is {best_n_estimator}")

    # get validation curve
    plt.figure()
    plt.plot(num_estimators, train_accuracy_data, "-", label="train")
    plt.plot(num_estimators, test_accuracy_data, "-", label="validation")
    plt.xlabel("number of estimators")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : number of estimators vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_adaboostedtree_validation.png")

    return best_n_estimator

def get_learning_curve(file_name, dataset_to_use, estimator=20):


    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = ada_boosted_tree(X_train, X_test, y_train, y_test, num_samples=num_samples, n_estimators=estimator)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure()
    plt.plot(train_samples_list, train_accuracy_data, "-", label="train")
    plt.plot(train_samples_list, test_accuracy_data, "-", label="test")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : training samples vs accuracy estimators={estimator}")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_adaboostedtree_learning.png")

if __name__ == "__main__":

    file_name = sys.argv[2]
    dataset_to_use = sys.argv[1]
    estimator = get_validation_curve(file_name, dataset_to_use)

    get_learning_curve(file_name, dataset_to_use, estimator=estimator)