from sklearn import decomposition
from scipy.stats import norm, kurtosis
from statistics import mean 
from sklearn import random_projection
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import helper


def apply_pca(X_train, X_test, y_train, y_test, n_components=2):

    """ Function used to apply PCA with num of components to data and return transformed train and test data. """


    print(f"\n\n Running PCA with n_components={n_components}")
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X_train)
    new_X_train = pca.transform(X_train)
    new_X_test = pca.transform(X_test)

    print(f"\nExplained Variance {pca.explained_variance_}")
    print(f"\nExplained Variance Ratio {pca.explained_variance_ratio_}")
    print(new_X_train.shape)
    print(new_X_test.shape)

    plt.figure()
    plt.plot(pca.explained_variance_, "-o")
    plt.xlabel("num components")
    plt.ylabel("Explained Variance (Largest Eigenvalue) ")
    plt.title(" Num of Components vs Explained Variance")
    # plt.legend(loc="upper left")
    plt.savefig(f"numcomponents_vs_explainedvariance_{dataset_to_use}.png")

    plt.figure()
    plt.plot(pca.explained_variance_ratio_, "-o")
    plt.xlabel("num components")
    plt.ylabel("Explained Variance Ratio ")
    plt.title(" Num of Components vs Explained Variance Ratio")
    # plt.legend(loc="upper left")
    plt.savefig(f"numcomponents_vs_explainedvariance_ratio_{dataset_to_use}.png")

    return  pca.explained_variance_

def apply_ICA(X_train, X_test, y_train, y_test, n_components=2):

    """ Function used to apply ICA with num of components to data and return transformed train and test data. """
    print(f"\n\n Running ICA with n_components={n_components}")
    ica = decomposition.FastICA(n_components=n_components, whiten=True)
    ica.fit(X_train)
    X_train = ica.transform(X_train)
    X_test = ica.transform(X_test)
  
    kurt = kurtosis(X_train)

    print(f"Kurtosis: {kurt}")
    print(f"AVG Kurtosis: {mean(abs(kurt))}")

    return  mean(abs(kurt))

def apply_random_projection(X_train, X_test, y_train, y_test, n_components=2):

    rand_proj = random_projection.GaussianRandomProjection(n_components=n_components, )
    rand_proj.fit(X_train)

    new_X_train = rand_proj.transform(X_train)
    new_X_test = rand_proj.transform(X_test)


    # reconstruct and get error
    inverse_data = np.linalg.pinv(rand_proj.components_.T)
  
    reconstructed_data = new_X_train.dot(inverse_data)

    df_reconstructed_data = pd.DataFrame(reconstructed_data)
    

    error = mean_squared_error(X_train, df_reconstructed_data)
    print(f"Mean Squared Error: {error}")
    return error

def feature_rediuction():
    pass

if __name__ == "__main__":
    
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)
    print(X_train.shape)


    #exp_variance = apply_pca(X_train, X_test, y_train, y_test, n_components=X_train.shape[1])
    avg_kurtosis = []

    #for i in range(1, X_train.shape[1]):
    #    k = apply_ICA(X_train, X_test, y_train, y_test, n_components=i)
    #    avg_kurtosis.append(k)
#
    #plt.figure()
    #plt.plot(avg_kurtosis, "-o")
    #plt.xlabel("num components")
    #plt.ylabel("Avg Kurtosis ")
    #plt.title(" Num of Components vs Avg Kurtosis")
    ## plt.legend(loc="upper left")
    #plt.savefig(f"numcomponents_vs_avg_kurtosis_{dataset_to_use}.png")
    ms_errors =[]
    for i in range(1, X_train.shape[1]):
        e = apply_random_projection(X_train, X_test, y_train, y_test, n_components=i)
        ms_errors.append(e)
    
    plt.figure()
    plt.plot(ms_errors, "-o")
    plt.xlabel("num components")
    plt.ylabel("Reconstruction Error")
    plt.title(" Num of Components vs Reconstruction Error")
    #plt.legend(loc="upper left")
    plt.savefig(f"numcomponents_vs_msqe_{dataset_to_use}.png")