import pandas as pd

from ImplementedMethods import ImplementedMethods, clusters_visualization
from DimensionReduction import DimensionReduction, get_reduced_data
from utils import metrics, to_0_1_range, save_data, get_true_labels


def run_dbscan_test(methods, error_col, reduction, visualization=True, params=None):
    print("DBscan without pretraining")
    model, x, labels, _, true_labels = methods.dbscan(error_col=error_col, pretraining=False,
                                                           visualization=visualization, reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("DBscan", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)


def run_dbscan_test_pretraining(methods, error_col, reduction, visualization=True, params=None):
    print("DBscan pretraining")
    model, x, labels, _, true_labels = methods.dbscan(error_col=error_col, pretraining=True,
                                                           visualization=visualization, reduction=reduction)
    print("Dbscan test after pretraining")

    test_data = methods.data.data_drive_scaled_err
    # added dimension reduction
    test_data = get_reduced_data(test_data[methods.data.cols], method=reduction, componants=3, scale=True,
                                 scale_fit=False)

    true_labels = get_true_labels(methods.data.data_drive_scaled_err, error_col)
    labels = model.predict(test_data).tolist()
    labels = to_0_1_range(labels, -1)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("DBscan", reduction, 'pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
    if visualization:
        clusters_visualization(test_data, labels, model, "Dbscan test after pretraining")


def run_spectral_clustering_test(methods, error_col, visualization=True, params=None):
    x, labels, _, true_labels = methods.spectral_clustering(error_col=error_col)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)


def run_isolation_forests_test(methods, error_col, reduction, visualization=True, params=None):
    print("Isolation forests without pretraining")
    model, x, labels, _, true_labels = methods.isolation_forests(error_col=error_col, pretraining=False,
                                                                      visualization=visualization, reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Isolation forests", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)


def run_isolation_forests_test_pretraining(methods, error_col, reduction, visualization=True, params=None):
    print("Isolation forests pretraining")
    model, x, labels, _, true_labels = methods.isolation_forests(error_col=error_col,
                                                                      pretraining=False,
                                                                      visualization=visualization,
                                                                      reduction=reduction)
    print("Isolation forests test after pretraining")
    test_data = methods.data.data_drive_scaled_err
    # added dimension reduction
    test_data = get_reduced_data(test_data[methods.data.cols], method=reduction, componants=3, scale=True,
                                 scale_fit=False)

    true_labels = get_true_labels(methods.data.data_drive_scaled_err, error_col)
    labels = model.predict(test_data).tolist()
    labels = to_0_1_range(labels, -1)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Isolation forests", reduction, 'pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
    if visualization:
        clusters_visualization(test_data, labels, model, "Isolation forests test after pretraining")


def run_local_outlier_factor_test(methods, error_col, reduction, visualization=True, params=None):
    print("Local outlier factor without pretraining")
    model, x, labels, _, true_labels = methods.local_outlier_factor(error_col=error_col,
                                                                         pretraining=False,
                                                                         visualization=visualization,
                                                                         reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Local outlier factor", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)


def run_local_outlier_factor_test_pretraining(methods, error_col, reduction, visualization=True, params=None):
    print("Local outlier factor pretraining")
    model, x, labels, _, true_labels = methods.local_outlier_factor(error_col=error_col,
                                                                         pretraining=False,
                                                                         visualization=visualization,
                                                                         reduction=reduction)
    print("Local outlier factor test after pretraining")
    test_data = methods.data.data_drive_scaled_err
    # added dimension reduction
    test_data = get_reduced_data(test_data[methods.data.cols], method=reduction, componants=3, scale=True,
                                 scale_fit=False)

    true_labels = get_true_labels(methods.data.data_drive_scaled_err, error_col)

    labels = model.predict(test_data).tolist()
    labels = to_0_1_range(labels, -1)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Local outlier factor", reduction, 'pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
    if visualization:
        clusters_visualization(test_data, labels, model, "Local outlier factor test after pretraining")


def run_gaussian_mixture_test(methods, error_col, reduction, visualization=True, params=None):
    print("Gaussian mixture without pretraining")
    model, x, labels, _, true_labels = methods.gaussian_mixture(error_col=error_col,
                                                                     pretraining=False,
                                                                     visualization=visualization,
                                                                     reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Gaussian mixture", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)


def run_gaussian_mixture_test_pretraining(methods, error_col, reduction, visualization=True, params=None):
    print("Gaussian mixture pretraining")
    model, x, labels, _, true_labels = methods.gaussian_mixture(error_col=error_col,
                                                                     pretraining=False,
                                                                     visualization=visualization,
                                                                     reduction=reduction)
    print("Gaussian mixture test after pretraining")
    test_data = methods.data.data_drive_scaled_err
    # added dimension reduction
    test_data = get_reduced_data(test_data[methods.data.cols], method=reduction, componants=3, scale=True,
                                 scale_fit=False)

    true_labels = get_true_labels(methods.data.data_drive_scaled_err, error_col)

    labels = model.predict(test_data).tolist()
    labels = to_0_1_range(labels, 0)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("Gaussian mixture", reduction, 'pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
    if visualization:
        clusters_visualization(test_data, labels, model, "Gaussian mixture test after pretraining")
        
def run_knn_model(methods, error_col, reduction, visualization=True, params=None):
    print("KNN model")
    model, x, labels, _, true_labels = methods.KNN(error_col=error_col, pretraining=False,
                                                   visualization=visualization, reduction=reduction, neighbour_count=20)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("KNN", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
        
def run_svc_model(methods, error_col, reduction, visualization=True, params=None):
    print("SVC model")
    model, x, labels, _, true_labels = methods.SVC(error_col=error_col, pretraining=False,
                                                   visualization=visualization, reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("SVC", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
        
def run_decision_tree_model(methods, error_col, reduction, visualization=True, params=None):
    print("Decision Tree model")
    model, x, labels, _, true_labels = methods.DecisionTree(error_col=error_col, pretraining=False,
                                                            visualization=visualization, reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("DecisionTree", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)
        
def run_logistic_regression_model(methods, error_col, reduction, visualization=True, params=None):
    print("Logistic Regression model")
    model, x, labels, _, true_labels = methods.LogisticRegression(error_col=error_col, pretraining=False,
                                                                  visualization=visualization, reduction=reduction)
    accuracy, sensitivity, specificity, F1_score, AUC = metrics(true_labels, labels, visualization)
    print("Accuracy: ", accuracy, "Sensitivity: ", sensitivity, "Specificity: ", specificity, "F1_score: ", F1_score,
          "AUC: ", AUC)
    save_data("LogisticRegression", reduction, 'no pretraining', accuracy, sensitivity, specificity, F1_score, AUC, params=params)


def run_all_clustering_methods(methods, error_col, reduction='PCA', visualization=True, params=None):
    run_dbscan_test(methods, error_col, reduction, visualization=visualization, params=params)
    # run_dbscan_test_pretraining(methods, error_col, reduction, visualization=visualization, params=params)

    # run_spectral_clustering_test(methods, error_col, visualization=visualization, params=params)

    run_isolation_forests_test(methods, error_col, reduction, visualization=visualization, params=params)
    # run_isolation_forests_test_pretraining(methods, error_col, reduction, visualization=visualization, params=params)

    run_local_outlier_factor_test(methods, error_col, reduction, visualization=visualization, params=params)
    # run_local_outlier_factor_test_pretraining(methods, error_col, reduction, visualization=visualization, params=params)

    run_gaussian_mixture_test(methods, error_col, reduction, visualization=visualization, params=params)
    # run_gaussian_mixture_test_pretraining(methods, error_col, reduction, visualization=visualization, params=params)
    
    run_knn_model(methods, error_col, reduction, visualization=visualization, params=params)
    run_svc_model(methods, error_col, reduction, visualization=visualization, params=params)
    run_decision_tree_model(methods, error_col, reduction, visualization=visualization, params=params)
    run_logistic_regression_model(methods, error_col, reduction, visualization=visualization, params=params)
