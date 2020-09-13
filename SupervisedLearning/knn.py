import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import read_csv_data, split_data, format_phishing_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def k_nearest_neighbors(X, y, neighbors=3):
    print(f"\n :: Knn Classifier with {neighbors} neighbors")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(X, y)

    # predict values
    test_accuracy = accuracy_score(knn.predict(X_test), y_test)
    train_accuracy = accuracy_score(knn.predict(X_train), y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")


if __name__ == "__main__":

    dataset = read_csv_data(sys.argv[1], delimiter=",", encode=False)
    X, y = split_data(dataset, class_attr="class")

    X, y = format_phishing_data(sys.argv[1])

    for i in range(100):
        k_nearest_neighbors(X, y, neighbors=i+1)