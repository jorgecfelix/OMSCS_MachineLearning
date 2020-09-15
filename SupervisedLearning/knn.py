import sys
import numpy as np
import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def k_nearest_neighbors(X, y, neighbors=3, num_samples=100):
    print(f"\n :: Knn Classifier with {neighbors} neighbors")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # slice training data
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]

    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(X_train, y_train)

    # predict values
    test_accuracy = accuracy_score(knn.predict(X_test), y_test)
    train_accuracy = accuracy_score(knn.predict(X_train), y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")

    return test_accuracy

if __name__ == "__main__":

    # dataset = read_csv_data(sys.argv[1], delimiter=",", encode=False)
    # X, y = split_data(dataset, class_attr="class")

    X, y = helper.format_phishing_data(sys.argv[1])
    # X, y = helper.format_swarm_data(sys.argv[1])

    training_samples_list = np.arange(100, 3000, 50)

    test_accuracy_list = [k_nearest_neighbors(X, y, num_samples=num_samples) for num_samples in training_samples_list]

    helper.plot_accuracy_vs_training_samples(training_samples_list, [(test_accuracy_list, "knn")])
