from sklearn import mixture
import matplotlib.pyplot as plt
import dimreduction
import numpy as np
import helper
import sys

# NOTE: The code below was created by using gaussion mixture examples on scikitlearn's documention

def run_gmm(X_train, X_test, y_train, y_test, n_components=2, cv='full'):

    clf = mixture.GaussianMixture(n_components=n_components, covariance_type=cv, reg_covar=0.00001)
    clf.fit(X_train)
    
    bic = clf.bic(X_train)
    aic = clf.aic(X_train)

    print(f"\n\n Running GMM using num components {n_components} and cv type {cv}")
    print(f"BIC : {bic}")
    print(f"AIC : {aic}")
    return bic, aic

def run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='bic', dimreduc=''):    


    cv_types = ['spherical', 'tied', 'diag', 'full']
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

        pass
    elif dimreduc == 'rca':
        # choose number of components for pca
        if dataset_to_use == 'd1':
            n_components = 30
        elif dataset_to_use == 'd2':
            n_components = 11
        e, X_train, X_test = dimreduction.apply_random_projection(i_X_train, i_X_test, y_train, y_test, n_components=n_components)

        pass
    elif dimreduc == 'rfe':
        scores, X_train, X_test = dimreduction.apply_recursive_feature_elimination(i_X_train, i_X_test, y_train, y_test)

    else:
        X_train = i_X_train
        X_test = i_X_test

    bics = []
    aics = []
    ncomps = range(1, 21)

    plt.figure()

    for cv in cv_types:
        bics = []
        aics = []
        for n in ncomps:
            bic, aic = run_gmm(X_train, X_test, y_train, y_test, n_components=n, cv=cv)
            bics.append(bic)
            aics.append(aic)

        if plot_type == 'bic':
            plt.plot(ncomps, bics, "-o", label=cv)
        elif plot_type == 'aic':
            plt.plot(ncomps, aics, "-o", label=cv)

    if plot_type == 'bic':
        plt.xlabel("num components")
        plt.ylabel("bic")
        plt.title(" Num of Components vs BIC")
        plt.legend(loc="upper right")
        plt.savefig(f"expecmax_{dimreduc}_numcomps_vs_bic_{dataset_to_use}.png")
    elif plot_type == 'aic':
        plt.xlabel("num components")
        plt.ylabel("aic")
        plt.title(" Num of Components vs AIC")
        plt.legend(loc="upper right")
        plt.savefig(f"expecmax_{dimreduc}_numcomps_vs_aic_{dataset_to_use}.png")


if __name__ == "__main__":
    
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    for alg in ['', 'pca', 'ica', 'rca', 'rfe']:

       run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='bic', dimreduc=alg)
       run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='aic', dimreduc=alg)