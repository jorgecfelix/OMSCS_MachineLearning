import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import helper
from sklearn import svm

FIGURE_NUM = 0

def run_svm(X, y, num_samples=None, kernel='rbf', max_iter=10):
    print("\n :: SVM Classifier")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.4)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    svm_classifier = svm.SVC(kernel=kernel, max_iter=max_iter)
    svm_classifier.fit(X_train, y_train)

    # predict values
    test_accuracy = svm_classifier.score(X_test, y_test)
    train_accuracy = svm_classifier.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")
    
    return train_accuracy, test_accuracy

def get_validation_curve(file_name, dataset_to_use, kernel):

    X, y, train_samples_list = helper.get_dataset(dataset_to_use, file_name)


    max_iter = [x for x in range(1, 100)]
    test_accuracy_data = []
    train_accuracy_data = []

    for i in max_iter:
        train_accuracy, test_accuracy  = run_svm(X, y, max_iter=i, kernel=kernel)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)
    
    # get learning curves
    plt.figure()
    plt.plot(max_iter, train_accuracy_data, "-", label=f"train {kernel}")
    plt.plot(max_iter, test_accuracy_data, "-", label=f"test {kernel}")
    plt.xlabel("max iterations")
    plt.ylabel("accuracy")
    plt.title(f"{dataset_to_use} : max iterations vs accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_to_use}_svm_{kernel}_validation.png")


def get_learning_curve(file_name, dataset_to_use, kernel):
    X, y, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy  = run_svm(X, y, num_samples=num_samples, max_iter=20, kernel=kernel)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure()
    plt.plot(train_samples_list, train_accuracy_data, "-", label=f"train {kernel}")
    plt.plot(train_samples_list, test_accuracy_data, "-", label=f"test {kernel}")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : number training samples vs accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_to_use}_svm_{kernel}_learning.png")

if __name__ == "__main__":
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    for kernel in ['rbf', 'sigmoid']:
        get_validation_curve(file_name, dataset_to_use, kernel=kernel)
        get_learning_curve(file_name, dataset_to_use,  kernel=kernel)