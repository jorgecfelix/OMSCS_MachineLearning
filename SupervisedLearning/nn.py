import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import read_csv_data, split_data, format_phishing_data
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)



def neural_net(X, y):
    print("\n :: Neural Net Classifier")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # number of attributs
    num_attr = X.shape[1]
    print(num_attr)

    print(X.shape, y.shape)
    # create model
    model = keras.Sequential([
    keras.layers.Input(shape=(num_attr)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='relu')
    ])

    print(model.summary())
    # compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, epochs=20)
    
    print("\n Test and Train Accuracy below")
    print(model.evaluate(X_test, y_test))
    print(model.evaluate(X_train, y_train))

if __name__ == "__main__":

    dataset = read_csv_data(sys.argv[1], delimiter=",", encode=False)

    X, y = split_data(dataset, class_attr="class")

    X, y = format_phishing_data(sys.argv[1], is_nn=True)

    neural_net(X, y)