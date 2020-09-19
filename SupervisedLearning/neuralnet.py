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



def neural_net(X_train, X_test, y_train, y_test, num_samples=None, epochs=15, learning_rate=0.001):
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
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt,
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

    learning_rates = np.arange(0.0001, 0.0005, 0.0001)
    test_accuracy_data = []
    train_accuracy_data = []

    for lr in learning_rates:
        train_accuracy, test_accuracy  = neural_net(X_train, X_test, y_train, y_test, learning_rate=lr)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get best neighbors to use
    best_lr = learning_rates[test_accuracy_data.index(max(test_accuracy_data))]
    
    print(f"\n :: Neural Net :: Best learning rate to use is {best_lr}")

    plt.figure()
    plt.plot(learning_rates, train_accuracy_data, "-", label="train")
    plt.plot(learning_rates, test_accuracy_data, "-", label="validation")
    plt.xlabel("learning rate")
    plt.ylabel("accuracy")
    plt.title(f"{dataset_to_use} : learning rate vs accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_to_use}_neuralnet_validation.png")

    return best_lr

def get_learning_curve(file_name, dataset_to_use, learning_rate=0.001):
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = neural_net(X_train, X_test, y_train, y_test, num_samples=num_samples, learning_rate=learning_rate)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get learning curves
    plt.figure()
    plt.plot(train_samples_list, train_accuracy_data, "-", label="train")
    plt.plot(train_samples_list, test_accuracy_data, "-", label="test")
    plt.xlabel("number of samples")
    plt.ylabel("accuracy ")
    plt.title(f"{dataset_to_use} : training samples vs accuracy learning_rate={learning_rate}")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_to_use}_neuralnet_learning.png")

if __name__ == "__main__":
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    lr = get_validation_curve(file_name, dataset_to_use)
    get_learning_curve(file_name, dataset_to_use, learning_rate=lr)

