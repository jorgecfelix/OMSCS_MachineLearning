import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_csv_data(file_name, delimiter=","):
    """ 
        Function used to read in csv data into a pandas dataframe
    """

    dataset = pd.read_csv(file_name, delimiter=delimiter)
    
    # Show dataframe information
    print(dataset.info())

    return dataset

        
def split_data(dataset, class_attr='class'):
    """ Function used to split class and attribute data into respective dataframes."""

    X = dataset.drop([class_attr], axis=1)
    y = dataset[class_attr]
    
    print( "\n Number of values per class attribute:")
    print(y.value_counts())

    # return attributes and labels
    return X, y


def get_accuracy(actual, preds):
    """ Helper function used to count wrong predictions and getting accuracy"""
    count = 0
    for i in range(len(actual)):
        if actual.iloc[i] != preds[i]:
            count += 1
    
    return 1.0 - (count / len(actual))

def plot_accuracy_vs_training_samples(num_samples, accuracy_data):
    """ Function used to plot testing data accuracy vs number of samples used for training.""" 
  
    plt.plot(num_samples, accuracy_data, "-o", label="test")
    plt.xlabel("Num of Training Samples")
    plt.ylabel("Accuracy ")
    plt.legend(loc="upper left")
    plt.show()