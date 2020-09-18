import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import helper


print(tf.__version__)



def neural_net(X_train, X_test, y_train, y_test, num_samples=None, epochs=15):
    print("\n :: Neural Net Classifier")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.4)

    # slice training data
    if num_samples != None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    print( "\n Number of values per class attribute used to Train:")
    print(y_train.value_counts())
    print( "\n Number of values per class attribute used to Test:")
    print(y_test.value_counts())

    # number of attributs
    num_attr = X_train.shape[1]
    print(num_attr)

    print(X_train.shape, y_train.shape)
    # create model
    model = keras.Sequential([
    keras.layers.Input(shape=(num_attr)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])

    # print(model.summary())
    # compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, epochs=epochs)


    print(f" Number of training samples {X_train.shape[0]}")
    print("\n Test and Train Accuracy below")
    print(model.evaluate(X_test, y_test))
    print(model.evaluate(X_train, y_train))

    return model.evaluate(X_train, y_train)[1], model.evaluate(X_test, y_test)[1]


def get_validation_curve(file_name, dataset_to_use):

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    # split training for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.40)

    epochs = [x for x in range(1, 50, 5)]
    test_accuracy_data = []
    train_accuracy_data = []

    for epoch in epochs:
        train_accuracy, test_accuracy  = neural_net(X_train, X_test, y_train, y_test, epochs=epoch)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    plt.figure()
    plt.plot(epochs, train_accuracy_data, "-", label="train")
    plt.plot(epochs, test_accuracy_data, "-", label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"{dataset_to_use} : epochs vs accuracy")
    plt.legend(loc="upper left")
    plt.savefig(f"{dataset_to_use}_neuralnet_validation.png")


def get_learning_curve(file_name, dataset_to_use):
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = neural_net(X_train, X_test, y_train, y_test, num_samples=num_samples)
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
    plt.savefig(f"{dataset_to_use}_neuralnet_learning.png")

if __name__ == "__main__":
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    get_validation_curve(file_name, dataset_to_use)
    get_learning_curve(file_name, dataset_to_use)

