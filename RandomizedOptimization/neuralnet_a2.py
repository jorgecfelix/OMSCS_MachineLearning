import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np
import helper


def neural_net(X_train, X_test, y_train, y_test, num_samples=None, epochs=15, learning_rate=0.001, algorithm="gradient_descent"):
    print("\n :: Neural Net Classifier")


    # slice training data if num_samples is None
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

    # Initialize neural network object and fit object
    model = mlrose.NeuralNetwork(hidden_nodes = [4, 4], activation = 'sigmoid', \
                                     algorithm = algorithm, max_iters = 100, \
                                     bias = True, is_classifier = True, learning_rate = learning_rate, \
                                     early_stopping = True, clip_max = 5, max_attempts = 10, \
                                     random_state = 3)
    
    model.fit(X_train, y_train)

    # get accuracy on train data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # get accuracy on test data
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    

    print(f" Number of training samples used {X_train.shape[0]}")
    print("\n Test and Train Accuracy below")
    print(f"Test Accuracy => {test_accuracy}")
    print(f"Train Accuracy => {train_accuracy}")

    return train_accuracy, test_accuracy

def get_validation_curve(file_name, dataset_to_use, algorithm='gradient_descent'):

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    # split training for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.40)

    learning_rates = [0.000001, 0.0000025, 0.000005, 0.0000075,
                      0.00001, 0.000025, 0.00005, 0.000075,
                      0.0001, 0.00025, 0.0005, 0.00075,
                      0.001, 0.0025, 0.005, 0.0075,
                      0.01, 0.025, 0.05, 0.075,
                      0.1, 0.25, 0.5, 0.75, 
                      1.0]#np.arange(0.0001, 0.005, 0.0001)

    test_accuracy_data = []
    train_accuracy_data = []

    for lr in learning_rates:
        print(f"\n\n :: Neural Net :: Using learning rate =>  {lr}")
        train_accuracy, test_accuracy  = neural_net(X_train, X_test, y_train, y_test, learning_rate=lr, algorithm=algorithm)
        test_accuracy_data.append(test_accuracy)
        train_accuracy_data.append(train_accuracy)

    # get best neighbors to use
    best_lr = learning_rates[test_accuracy_data.index(max(test_accuracy_data))]
    
    print(f"\n :: Neural Net :: Best learning rate to use is {best_lr}")
    lr_index = list(range(len(learning_rates)))

    plt.figure()
    plt.plot(lr_index, train_accuracy_data, "-o", label="train")
    plt.plot(lr_index, test_accuracy_data, "-o", label="validation")

    plt.xticks(lr_index, learning_rates)
    plt.locator_params(axis='x', nbins=8)

    plt.xlabel("learning rate")
    plt.ylabel("accuracy")
    plt.title(f"{dataset_to_use} : learning rate vs accuracy")
    plt.legend(loc="lower right")
    plt.savefig(f"{dataset_to_use}_neuralnet_validation_{algorithm}_a2.png")

    return best_lr

def get_learning_curve(file_name, dataset_to_use, learning_rate=0.001, algorithm='gradient_descent'):
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    
    test_accuracy_data = []
    train_accuracy_data = []

    # using best ccp_alpha train decision tree on different number of samples
    for num_samples in train_samples_list:
        print(f'\n Number of training samples used => {num_samples}')
        train_accuracy, test_accuracy = neural_net(X_train, X_test, y_train, y_test, num_samples=num_samples, learning_rate=learning_rate, algorithm=algorithm)
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
    plt.savefig(f"{dataset_to_use}_neuralnet_learning_{algorithm}_a2.png")

if __name__ == "__main__":
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]
    
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)
    train_accuracy, test_accuracy = neural_net(X_train, X_test, y_train, y_test, num_samples=None, learning_rate=0.001)

    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']
    for alg in algorithms:
       lr = get_validation_curve(file_name, dataset_to_use, algorithm=alg)
       get_learning_curve(file_name, dataset_to_use, learning_rate=lr, algorithm=alg)