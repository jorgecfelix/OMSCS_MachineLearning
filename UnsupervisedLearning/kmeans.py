from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import helper
import dimreduction


# NOTE: The code below was written by Jorge Felix using scikitlearn's examples as a reference

def run_kmeans(X_train, X_test, y_train, y_test, n_clusters=2):

    kmeans = KMeans(n_clusters=n_clusters, random_state=1, algorithm='full').fit(X_train)

    print(f"\n\n Number of clusters {n_clusters}")
    #print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    print(f" Inertia: {kmeans.inertia_}")


    return kmeans.inertia_

def get_inertia_plot_nodimreduc(dataset_to_use, file_name, dimreduc=''):

    i_X_train, i_X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)

    
    if dimreduc == 'pca':
        # choose number of components for pca
        if dataset_to_use == 'd1':
            n_components = 5
        elif dataset_to_use == 'd2':
            n_components = 2

        # apply pca
        exp_variance, X_train, X_test = dimreduction.apply_pca(i_X_train, i_X_test, y_train, y_test, n_components=n_components)

    elif dimreduc == 'ica':
        # choose number of components for pca
        if dataset_to_use == 'd1':
            n_components = 30
        elif dataset_to_use == 'd2':
            n_components = 2

        k, X_train, X_test = dimreduction.apply_ICA(i_X_train, i_X_test, y_train, y_test, n_components=n_components)

    elif dimreduc == 'rca':
        # choose number of components for pca
        if dataset_to_use == 'd1':
            n_components = 20
        elif dataset_to_use == 'd2':
            n_components = 9
        e, X_train, X_test = dimreduction.apply_random_projection(i_X_train, i_X_test, y_train, y_test, n_components=n_components)

    elif dimreduc == 'rfe':
        scores, X_train, X_test = dimreduction.apply_recursive_feature_elimination(i_X_train, i_X_test, y_train, y_test)

    else:
        X_train = i_X_train
        X_test = i_X_test

    print(f"\n\n New Shape {X_train.shape[1]}")
    inertias = []
    num_c = range(1,i_X_train.shape[1])
    for n in num_c:
        inertia = run_kmeans(X_train, X_test, y_train, y_test, n_clusters=n)
        inertias.append(inertia)

    
    return inertias


if __name__ == "__main__":
    
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]


    for alg in ['', 'pca', 'ica', 'rca', 'rfe']:

        inertias = get_inertia_plot_nodimreduc(dataset_to_use, file_name, dimreduc=alg)
        plt.figure()
        plt.plot(range(1, len(inertias) + 1), inertias, "-o", label=alg)    
        plt.xlabel("num clusters")
        plt.ylabel("Inertia ")
        plt.title(" Num of Clusters vs Inertia")
        plt.legend(loc="upper right")
        plt.savefig(f"kmeans_{alg}_numcluster_vs_inertia_{dataset_to_use}.png")