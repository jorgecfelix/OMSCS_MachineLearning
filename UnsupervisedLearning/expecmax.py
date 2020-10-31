from sklearn import mixture
import matplotlib.pyplot as plt
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

def run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='bic'):    

    cv_types = ['spherical', 'tied', 'diag', 'full']
    X_train, X_test, y_train, y_test, train_samples_list = helper.get_dataset(dataset_to_use, file_name, is_nn=True)
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
        plt.savefig(f"numcomps_vs_bic_{dataset_to_use}.png")
    elif plot_type == 'aic':
        plt.xlabel("num components")
        plt.ylabel("aic")
        plt.title(" Num of Components vs AIC")
        plt.legend(loc="upper right")
        plt.savefig(f"numcomps_vs_aic_{dataset_to_use}.png")


if __name__ == "__main__":
    
    dataset_to_use = sys.argv[1]
    file_name = sys.argv[2]

    run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='bic')
    run_gmm_nodimreduc(dataset_to_use, file_name, plot_type='aic')