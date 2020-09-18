import sys
import numpy as np
import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def k_nearest_neighbors(X_train, X_test, y_train, y_test, neighbors=3, num_samples=None):
    print(f"\n :: Knn Classifier with {neighbors} neighbors")

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.40)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    print( "\n Number of values per class attribute used to Train:")
    print(y_train.value_counts())
    print( "\n Number of values per class attribute used to Test:")
    print(y_test.value_counts())

    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(X_train, y_train)

    # predict values
    test_accuracy = accuracy_score(knn.predict(X_test), y_test)
    train_accuracy = accuracy_score(knn.predict(X_train), y_train)

    print(f" Number of training samples {X_train.shape[0]}")
    print(f" Number of test samples {X_test.shape[0]}")
    print(f" Accuracy of test data : {test_accuracy}")
    print(f" Accuracy of train data : {train_accuracy}\n")

    return train_accuracy, test_accuracy 

def get_validation_curve(file_name, dataset_to_use):

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    
    neighbors_list = [x for x in range(1, 31)]
    test_accuracy_data = []
    train_accuracy_data = []

    for neighbor in neighbors_list:
        train_accuracy, test_accuracy  = k_nearest_neighbors(X_train, X_test, y_train, y_test, neighbors=neighbor)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)
    # get learning curves
    plt.figure()
    plt.plot(neighbors_list, train_accuracy_data, "-", label="train")
    plt.plot(neighbors_list, test_accuracy_data, "-", label="test")
    plt.xlabel("number of neigbors")
    plt.ylabel("accuracy")
    plt.title(f"{dataset_to_use} : number of neighbors vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_knn_validation.png")


def get_learning_curve(file_name, dataset_to_use):
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = k_nearest_neighbors(X_train, X_test, y_train, y_test, num_samples=num_samples, neighbors=5)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure()
    plt.plot(train_samples_list, train_accuracy_data, "-", label="train")
    plt.plot(train_samples_list, test_accuracy_data, "-", label="test")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : number training samples vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_knn_learning.png")

if __name__ == "__main__":
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    get_validation_curve(file_name, dataset_to_use)
    get_learning_curve(file_name, dataset_to_use)
