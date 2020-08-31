import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class SupervisedLearning:
    """ 
        This class contains all the code for the Supervised Learning assignment

    """
    def __init__(self):
        # decision tree object
        # self.decison_tree = None 
        pass

    def read_csv_data(self, file_name):
        """ 
            Function used to read in csv data into a pandas dataframe
        """

        self.dataset = pd.read_csv(file_name, delimiter=";")

        print(self.dataset)
        
        # separate the attributes x, and class y from dataset
    def split_data(self):
        """ Function used to split class and attribute data into respective dataframes."""

        X = self.dataset.drop(['quality'], axis=1)
        y = self.dataset['quality']
        # print(x)
        # print(y)
        return X, y
    
    def run_decision_tree(self, test_size):
        """ Function used to fit decision tree and implement Complex Post Pruning"""
        X,y = self.split_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # create decision tree classifier and fit to training data
        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(X_train, y_train)
        
        # make predicitions
        # y_pred = self.decision_tree.predict(X_test)
        # y_train_preds = self.decision_tree.predict(X_train)
        
        test_accuracy = self.decision_tree.score(X_test, y_test)
        train_accuracy = self.decision_tree.score(X_train, y_train)

        print(f" Number of training samples {X_train.shape[0]}")
        print(f" Accuracy of test data : {test_accuracy}")
        print(f" Accuracy of train data : {train_accuracy}\n")


        return X_train.shape[0], test_accuracy, train_accuracy

    def get_accuracy(self, actual, preds):

        count = 0
        for i in range(len(actual)):
            if actual.iloc[i] != preds[i]:
                count += 1
        
        return 1.0 - (count / len(actual))

    def plot_decision_tree(self):
        _, ax = plt.subplots(figsize=(10, 10))  # whatever size you want
        tree.plot_tree(self.decision_tree, ax=ax)
        plt.show()

    def plot_accuracy_vs_training_samples(self, num_samples, accuracy_data):
        """ Function used to plot testing data accuracy vs number of samples used for training."""
        
        plt.plot(num_samples, accuracy_data, "-o", label="test")
        plt.xlabel("Num of Training Samples")
        plt.ylabel("Accuracy ")
        plt.legend(loc="upper left")

        plt.show()
        




if __name__ == "__main__":

    sl = SupervisedLearning()
    sl.read_csv_data(sys.argv[1])
    test_sizes = np.arange(.95, .05, -.05)
    
    num_samples_list = []
    accuracy_data = []
    for test_size in test_sizes:
        num_samples, test_accuracy, train_accuracy = sl.run_decision_tree(test_size)
        num_samples_list.append(num_samples)
        accuracy_data.append(test_accuracy)

    sl.plot_accuracy_vs_training_samples(num_samples_list, accuracy_data)