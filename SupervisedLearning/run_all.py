import sys
import neuralnet
import decisiontree
import adaboostedtree
import knn
import svm


if __name__ == "__main__":

    file_name = sys.argv[2]
    dataset_to_use = sys.argv[1]
    
    # run decision tree
    decisiontree.get_validation_and_learning_curve(file_name, dataset_to_use)

    # run adaboosted tree
    estimator = adaboostedtree.get_validation_curve(file_name, dataset_to_use)
    adaboostedtree.get_learning_curve(file_name, dataset_to_use, estimator=estimator)

    # run knn
    neighbors = knn.get_validation_curve(file_name, dataset_to_use)
    knn.get_learning_curve(file_name, dataset_to_use, neighbors=neighbors)

    #run svm
    for kernel in ['rbf', 'sigmoid']:
        iterations = svm.get_validation_curve(file_name, dataset_to_use, kernel=kernel)
        svm.get_learning_curve(file_name, dataset_to_use,  kernel=kernel, iterations=iterations)

    # run neural net
    lr = neuralnet.get_validation_curve(file_name, dataset_to_use)
    neuralnet.get_learning_curve(file_name, dataset_to_use, learning_rate=lr)

