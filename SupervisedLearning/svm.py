import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import helper
from sklearn import svm

def run_svm(X, y, num_samples=100, kernel='rbf'):
    print("\n :: SVM Classifier")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # slice training data
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    svm_classifier = svm.SVC(kernel=kernel)
    svm_classifier.fit(X_train, y_train)

    # predict values
    test_accuracy = svm_classifier.score(X_test, y_test)
    train_accuracy = svm_classifier.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")
    
    return test_accuracy

if __name__ == "__main__":

    # dataset = helper.read_csv_data(sys.argv[1], delimiter=",", encode=False)
    # X, y = helper.split_data(dataset, class_attr="class")

    X, y = helper.format_phishing_data(sys.argv[1])

    training_samples_list = np.arange(100, 3000, 50)


    test_accuracy_list = [run_svm(X, y, num_samples=num_samples) for num_samples in training_samples_list]

    helper.plot_accuracy_vs_training_samples(training_samples_list, [(test_accuracy_list, "svm")])
