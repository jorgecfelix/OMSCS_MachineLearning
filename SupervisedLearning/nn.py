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



def neural_net(X, y, num_samples=100):
    print("\n :: Neural Net Classifier")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # slice training data
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]

    # number of attributs
    num_attr = X.shape[1]
    print(num_attr)

    print(X.shape, y.shape)
    # create model
    model = keras.Sequential([
    keras.layers.Input(shape=(num_attr)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())
    # compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, epochs=10)


    print(f" Number of training samples {X_train.shape[0]}")
    print("\n Test and Train Accuracy below")
    print(model.evaluate(X_test, y_test))
    print(model.evaluate(X_train, y_train))
    
    return model.evaluate(X_test, y_test)[1]

if __name__ == "__main__":

    # dataset = helper.read_csv_data(sys.argv[1], delimiter=",", encode=False)

    # X, y = helper.split_data(dataset, class_attr="class")

    X, y = helper.format_phishing_data(sys.argv[1], is_nn=True)
    # X, y = helper.format_bank_data(sys.argv[1])
    training_samples_list = np.arange(100, X.shape[0], 50)

    test_accuracy_list = [neural_net(X, y, num_samples=num_samples) for num_samples in training_samples_list]

    helper.plot_accuracy_vs_training_samples(training_samples_list, [(test_accuracy_list, "neural network")])
