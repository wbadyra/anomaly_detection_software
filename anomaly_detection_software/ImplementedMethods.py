import copy
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import sys

from DataPreparation import DataPreparation
from DimensionReduction import DimensionReduction, get_reduced_data
from utils import to_0_1_range, get_true_labels


# optimized version of prediction - without checking duplicated points
def dbscan_predict(self, X_new):
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new.values
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    ind = np.reshape(self.core_sample_indices_, (-1, 1))
    comp_labels = np.hstack((self.components_, ind))
    # remove duplicated centre points
    p = [range(0, len(X_new[0]))]
    comp_labels[:, [range(0, len(X_new[0]))]]
    comp_labels = comp_labels[np.unique(comp_labels[:, [range(0, len(X_new[0]))]], return_index=True, axis=0)[1]]
    # matrix of distances between all grid points and cluster centres
    dist = cdist(X_new, comp_labels[:, 0:len(X_new[0])], 'euclidean')

    for j in range(len(dist)):
        # Find a core sample closer than EPS
        if len(dist[j] > 0) and min(dist[j]) < self.eps:
            i = np.argmin(dist[j])
            y_new[j] = self.labels_[int(comp_labels[i][len(X_new[0])])]

    return y_new


def ndim_grid(start, stop, n_points=100):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.linspace(start[i], stop[i], n_points) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0, 1).reshape(ndims, -1).T


def plot_2d(x_b, labels_clusters, x, labels, title):
    if isinstance(x, pd.DataFrame):
        x = x.values
    # Plot the clusters
    plt.scatter(x_b[:, 0], x_b[:, 1], c=labels_clusters, cmap="gray", s=0.4)  # plotting the clusters
    plt.colorbar()

    # Plot the clusters
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="rainbow", s=1)  # plotting the clusters
    plt.colorbar()
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.show()  # showing the plot


def plot_3d(x_b, labels_clusters, x, labels, title):
    # Plot the clusters - background
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_b[:, 0], x_b[:, 1], x_b[:, 2], c=labels_clusters, cmap="gray", s=0.4)  # plotting the clusters

    # Plot the clusters - data
    ax.scatter(x[:, 0], x[:, 1], x_b[:, 2], c=labels, cmap="plasma", s=1)  # plotting the clusters
    plt.title(title)
    plt.show()  # showing the plot


def clusters_visualization(x, labels, model, name='Model'):
    # construct array of background points for visualization
    x_b = ndim_grid(x.min(axis=0), x.max(axis=0), n_points=100)

    labels_clusters = model.predict(x_b)
    labels_clusters = to_0_1_range(labels_clusters, -1)
    plot_2d(x_b, labels_clusters, x, labels, name)


class ImplementedMethods:
    """Class containing fault detection methods based on clustering

            Attributes
            ----------
            data : DataPreparation
                Object containing a dataset, preprocessed in different ways'


            Methods
            -------
            dbscan():
                Density-Based Spatial Clustering
            spectral_clustering():
                Method clustering normalized Laplacian
            isolation_forests():
                This method isolates elements by random selection of split value
            local_outlier_factor():
                Result values for each point measure local deviation of the density of a given sample with respect to
                its neighbours
            gaussian_mixture():
                Clustering based on Gaussian probability distribution
            KNN():
                K-nearest neighbors model
            SVC():
                Support Vector Classifier
            DecisionTree():
                Model utilizing decision trees
            LogisticRegression():
                Classifier based on probability of predicted discrete input
            

    """

    def __init__(self, error_type=None, filename=None, error_cols=None):
        self.data = DataPreparation(filename=filename)
        self.data.filter_out_statinary_and_drive_data()
        self.data.insert_error(error_type, errors_col=error_cols)       
        self.data.scale_all_data()
        # self.data.visualize_data_error_comparison(['Engine RPM [RPM]'])

        print(self.data.data_scaled)
        print(self.data.data.describe())
        print(self.data.data_scaled.describe())

    def dbscan(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        if not pretraining:
            source_data = self.data.data_drive_scaled_err
        else:
            source_data = self.data.data_drive_scaled

        params = self.data.cols

        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)

        true_labels = get_true_labels(source_data, error_col)
        DBSCAN.predict = dbscan_predict
        model = DBSCAN(eps=0.05, min_samples=10)

        dbscan = model.fit(x)  # fitting the model
        labels = dbscan.labels_  # getting the labels
        np.set_printoptions(threshold=sys.maxsize)

        labels = to_0_1_range(labels, -1)

        if visualization:
            clusters_visualization(x, labels, model, 'DBscan')

        return dbscan, x, labels, params, true_labels

    def isolation_forests(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        # takes long time for more than 2 columns
        if not pretraining:
            source_data = self.data.data_drive_scaled_err
        else:
            source_data = self.data.data_drive_scaled

        params = self.data.cols

        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)

        true_labels = get_true_labels(source_data, error_col)

        clf = IsolationForest(random_state=0, n_estimators=1000, contamination='auto')
        labels = clf.fit_predict(x)
        np.set_printoptions(threshold=sys.maxsize)

        labels = to_0_1_range(labels, -1)

        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, clf, 'Isolation forests')

        return clf, x, labels, params, true_labels

    def local_outlier_factor(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        if not pretraining:
            source_data = self.data.data_drive_scaled_err
        else:
            source_data = self.data.data_drive_scaled

        params = self.data.cols

        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)

        true_labels = get_true_labels(source_data, error_col)

        clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
        clf.fit(x)
        labels = clf.predict(x)
        np.set_printoptions(threshold=sys.maxsize)

        labels = to_0_1_range(labels, -1)

        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, clf, 'Local outlier factor')

        return clf, x, labels, params, true_labels

    def gaussian_mixture(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        if not pretraining:
            source_data = self.data.data_drive_scaled_err
        else:
            source_data = self.data.data_drive_scaled

        params = self.data.cols

        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)

        true_labels = get_true_labels(source_data, error_col)

        model = GaussianMixture(n_components=3, random_state=0, covariance_type='tied')
        model.fit(x)  # fitting the model
        labels = model.predict(x)
        np.set_printoptions(threshold=sys.maxsize)

        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, model, 'Gaussian mixture')

        return model, x, labels, params, true_labels
    
    # classical algorithms
    
    def KNN(self, error_col, pretraining=False, visualization=False, reduction='PCA', neighbour_count=9):
        source_data = self.data.data_drive_scaled_err
            
        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)
        true_labels = get_true_labels(source_data, error_col)
        
        x_train, x_test, y_train, y_test = train_test_split(x, true_labels, test_size=0.2, stratify=true_labels)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = KNeighborsClassifier(n_neighbors=neighbour_count)
        
        model.fit(x_train, y_train)
        labels = model.predict(x_test)

        np.set_printoptions(threshold=sys.maxsize)
        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, model, 'KNN')

        return model, x, labels, self.data.cols, y_test
    
    def SVC(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        source_data = self.data.data_drive_scaled_err
            
        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)
        true_labels = get_true_labels(source_data, error_col)
        
        x_train, x_test, y_train, y_test = train_test_split(x, true_labels, test_size=0.2, stratify=true_labels)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = SVC(C=1, kernel='rbf', degree=3)
        
        model.fit(x_train, y_train)
        labels = model.predict(x_test)

        np.set_printoptions(threshold=sys.maxsize)
        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, model, 'KNN')

        return model, x, labels, self.data.cols, y_test
    
    def DecisionTree(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        source_data = self.data.data_drive_scaled_err
            
        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)
        true_labels = get_true_labels(source_data, error_col)
        
        x_train, x_test, y_train, y_test = train_test_split(x, true_labels, test_size=0.2, stratify=true_labels)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = DecisionTreeClassifier(random_state=0)
        
        model.fit(x_train, y_train)
        labels = model.predict(x_test)

        np.set_printoptions(threshold=sys.maxsize)
        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, model, 'KNN')

        return model, x, labels, self.data.cols, y_test
    
    def LogisticRegression(self, error_col, pretraining=False, visualization=False, reduction='PCA'):
        source_data = self.data.data_drive_scaled_err
            
        x = get_reduced_data(source_data[self.data.cols], method=reduction, componants=3, scale=True, scale_fit=True)
        true_labels = get_true_labels(source_data, error_col)
        
        x_train, x_test, y_train, y_test = train_test_split(x, true_labels, test_size=0.2, stratify=true_labels)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = LogisticRegression(random_state=0)
        
        model.fit(x_train, y_train)
        labels = model.predict(x_test)

        np.set_printoptions(threshold=sys.maxsize)
        # visualize cluster area
        if visualization:
            clusters_visualization(x, labels, model, 'KNN')

        return model, x, labels, self.data.cols, y_test