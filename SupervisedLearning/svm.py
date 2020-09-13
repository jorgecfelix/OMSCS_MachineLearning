import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import read_csv_data, split_data, format_phishing_data
from sklearn import svm

def run_svm(X, y):
    print("\n :: SVM Classifier")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # predict values
    test_accuracy = svm_classifier.score(X_test, y_test)
    train_accuracy = svm_classifier.score(X_train, y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")


if __name__ == "__main__":

    dataset = read_csv_data(sys.argv[1], delimiter=",", encode=False)
    X, y = split_data(dataset, class_attr="class")

    X, y = format_phishing_data(sys.argv[1])
    run_svm(X, y)