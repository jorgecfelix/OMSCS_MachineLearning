from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import helper


# NOTE: The code below was written by Jorge Felix using scikitlearn's examples as a reference

def run_kmeans(X_train, X_test, y_train, y_test, n_clusters=2):

    kmeans = KMeans(n_clusters=n_clusters, random_state=1, algorithm='full').fit(X_train)

    print(f"\n\n Number of clusters {n_clusters}")
    #print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    print(f" Inertia: {kmeans.inertia_}")


    return kmeans.inertia_

def get_inertia_plot_nodimreduc(dataset_to_use, file_name):

    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)
    inertias = []
    num_c = range(1,36)
    for n in num_c:
        inertia = run_kmeans(X_train, X_test, y_train, y_test, n_clusters=n)
        inertias.append(inertia)

    plt.figure()
    plt.plot(num_c, inertias, "-o")
    plt.xlabel("num clusters")
    plt.ylabel("Inertia ")
    plt.title(" Num of Clusters vs Inertia")
    # plt.legend(loc="upper left")
    plt.savefig(f"numcluster_vs_inertia_{dataset_to_use}.png")

if __name__ == "__main__":
    
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    get_inertia_plot_nodimreduc(dataset_to_use, file_name)