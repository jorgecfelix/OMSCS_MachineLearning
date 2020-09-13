import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_csv_data(file_name, delimiter=",", encode=False):
    """ 
        Function used to read in csv data into a pandas dataframe
    """

    dataset = pd.read_csv(file_name, delimiter=delimiter, header=None)

    if encode:
        dataset = encode_categorical_data(pd.DataFrame(dataset))

    # Show dataframe information
    print(dataset.info())
    
    # NOTE: Changing values to 0 and 1 df.loc[df['First Season'] > 1990, 'First Season'] = 1

    #dataset.loc[dataset[10] == 2, 10] = 0# = dataset[10].apply
    #dataset.loc[dataset[10] == 4, 10] = 1
    return dataset

def format_phishing_data(file_name, is_nn=False):
    """ Helper function used to format fishing data"""
    dataset = read_csv_data(file_name, delimiter=",", encode=False)
    
    # need to switch from -1,1 to 0,1 for binary classification
    if is_nn:
        dataset.loc[dataset[30] == -1, 30] = 0# = dataset[10].apply
        dataset.loc[dataset[30] == 1, 30] = 1

    print(dataset)
    # get all columns except last
    X = dataset.iloc[:, :-1]
    # get last column
    y = dataset.iloc[:,-1]

    print( "\n Number of values per class attribute:")
    print(y.value_counts())
    print(X)
    
    return X, y
    

def encode_categorical_data(dataset):

    # get categorical columns to encode
    categorical_cols = dataset.columns[dataset.dtypes==object].tolist()

    print(f"Categorical Columns retrieved => {categorical_cols}")
    
    # label encoder
    label_encoder = LabelEncoder()

    # apply le on categorical feature columns
    dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col: label_encoder.fit_transform(col))
    dataset[categorical_cols].head(10)

    return dataset
    

        
def split_data(dataset, class_attr='class'):
    """ Function used to split class and attribute data into respective dataframes."""

    #X = dataset.drop([class_attr], axis=1)
    #y = dataset[class_attr]
    print(dataset)
    X = dataset.drop(0, axis=1)

    X = X.iloc[:, :-1]

    y = dataset.iloc[:,-1]

    print( "\n Number of values per class attribute:")
    print(y.value_counts())
    print(X)
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