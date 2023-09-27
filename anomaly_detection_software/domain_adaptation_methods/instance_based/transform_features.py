import math
from random import sample
import numpy as np
import pandas as pd
from adapt.instance_based import WANN, BalancedWeighting, TrAdaBoost, KMM, KLIEP
import xgboost as xgb
import copy


def is_nan_in_lists(column_of_data):
    if None in column_of_data or any(math.isnan(x) for x in column_of_data.values):
        return True
    return False


def calculate_weights(source_feature, target_feature, source_label, target_label, algorithm, params=None):
    """Calculates weights given to each feature by DA algorithm.

    Args:
        source_feature (numpy.array): One dimensional array of source features.
        target_feature (numpy.array): One dimensional array of target features.
        source_label (List): List of source labels.
        target_label (List): List of target labels.
        algorithm (String): Selection of domain adaptation method.
        params (List): Optional, specific value of DA algorithm parameter (if present in algorithm's method call).

    Returns:
        Calculated weights for given source data
    """

    if algorithm == "TrAdaBoost":
        estimator = xgb.XGBClassifier()
        source_feature_reshaped = np.expand_dims(source_feature, axis=1)
        target_feature_reshaped = np.expand_dims(target_feature, axis=1)
        model = TrAdaBoost(estimator, n_estimators=100, Xt=target_feature_reshaped, yt=target_label, lr=1,
                           random_state=0, verbose=0)
        model.fit(source_feature_reshaped, source_label, target_feature_reshaped, target_label)
        return model.predict_weights(domain="src")

    y_source = np.asarray(source_label)
    x_source = np.asarray(source_feature)
    x_source = np.reshape(x_source, (-1, 1))

    y_target = np.asarray(target_label)
    x_target = np.asarray(target_feature)
    x_target = np.reshape(x_target, (-1, 1))

    if algorithm == "Balanced_Weighting":
        model = BalancedWeighting(xgb.XGBClassifier(), gamma=params, Xt=x_target, yt=y_target)
        _, x, y = model.fit_weights(x_source, x_target, y_source, y_target)
        return x[0:len(x_source)]

    if algorithm == "WANN":
        if params is not None:
            model = WANN(Xt=x_target, yt=y_target, random_state=0, verbose=1, C=params)
            model.fit(x_source, y_source, epochs=100, verbose=1)
        else:
            model = WANN(Xt=x_target, yt=y_target, C=0.8, random_state=0, verbose=3)
            model.fit(x_source, y_source, epochs=10, verbose=1)
        return model.predict_weights(x_source)
    elif algorithm == "KMM":
        if params is not None:
            model = KMM(xgb.XGBClassifier(), Xt=x_target, kernel="rbf", gamma=params, verbose=0, random_state=0)
            model.fit(x_source, y_source)
        else:
            model = KMM(xgb.XGBClassifier(), Xt=x_target, kernel="rbf", verbose=0, random_state=0)
            model.fit(x_source, y_source)
        return model.fit_weights(x_source, x_target)
    elif algorithm == "KLIEP":
        if params is not None:
            model = KLIEP(xgb.XGBClassifier(), Xt=x_target, kernel="rbf", gamma=[params/8, params, params*2],
                          random_state=0)
            model.fit(x_source, y_source)
        else:
            model = KLIEP(xgb.XGBClassifier(), Xt=x_target, kernel="rbf", gamma=[0.1, 1.], random_state=0)
            model.fit(x_source, y_source)
        return model.predict_weights(x_source)
    else:
        print("There is no such model implemented!")


def transform_featrure(weights, feature, algorithm=None):
    """Transforms given feature by provided weigths.

    Args:
        weights (numpy.array): One dimensional array of calculated weights.
        feature (numpy.array): One dimensional array of features.
        algorithm (String): Optional, DA algorithm name to provide, if special data format is needed.

    Returns:
        transformed (List): List of transformed features.
    """

    transformed = []
    feature = feature.tolist()
    for i in range(len(weights)):
        val = feature[i]*weights[i]
        # if WANN we need to [0]
        if algorithm == "WANN" or algorithm == "Balanced_Weighting":
            transformed.append(val[0])
        else:
            transformed.append(val)
    return transformed


def dataset_adaptation(train_dataset, algorithm, dataset_names, experiment, params=None):
    empty_columns = []
    for i in range(len(train_dataset)):
        for column in train_dataset[i].columns:
            if is_nan_in_lists(train_dataset[i][column]):
                print(column, dataset_names[i])
                if not column in empty_columns:
                    empty_columns.append(column)

    print(empty_columns)
    """Creates source and target datasets from selected datasets.

    Args:
        train_dataset (pandas.DataFrame): Training data for model.
        algorithm (String): Selection of domain adaptation method.
        dataset_names (List): List of dataset names which are included in experiment.
        experiment (String): Experiment name, saves results in proper destination.
        params (List): Optional, specific value of DA algorithm parameter (if present in algorithm's method call).

    Returns:
        train_dataset_adapted (pandas.DataFrame): Adapted training dataset.
    """

    train_dataset_adapted = []

    # check if modified datasets exist
    try:
        for i in range(len(train_dataset)):
            with open(f"/home/wasyl/projekt_badawczy/cbc_covid/adaptation_algorithms/modified_datasets/"
                      f"{experiment}/all_datasets/{dataset_names[i]}_{algorithm}_{params}.csv") as fileout:
                data_instance = pd.read_csv(fileout, index_col=0)
                data_instance = data_instance.reindex(sorted(data_instance.columns), axis=1)
                train_dataset_adapted.append(data_instance)

    # create modified datasets
    except FileNotFoundError:
        train_dataset_adapted = []
        for i in range(len(train_dataset)):
            train_temp_dataset = copy.deepcopy(train_dataset)
            source_dataset = copy.deepcopy(train_temp_dataset[i])
            target_dataset = copy.deepcopy(train_temp_dataset)
            target_dataframes = []
            for target in target_dataset:
                target_dataframes.append(target)
            target_dataset = pd.concat(target_dataframes)

            for column in source_dataset.columns.tolist():
                print("doing")
                # transform all features except target (label) and sex (categorical value)
                if column == 'target' or column == "Time":
                    continue
                weights_src = calculate_weights(source_dataset[column], target_dataset[column],
                                                source_dataset['target'],
                                                target_dataset['target'], algorithm=algorithm,
                                                params=params)

                transformed = transform_featrure(weights_src, source_dataset[column], algorithm=algorithm)
                transformed_series = pd.Series(transformed, name=column)

                # normalize transformed data
                transformed_series_norm = (transformed_series-transformed_series.min())/\
                                          (transformed_series.max()-transformed_series.min())
                train_temp_dataset[i].update(transformed_series_norm)
            # train_temp_dataset[i].to_csv(f"/home/wasyl/magisterka/fault_detection/processed_data/datasets/Dataset_after_da/{dataset_names[i]}.csv")
            train_dataset_adapted.append(train_temp_dataset[i])

    train_dataset_adapted = pd.concat(train_dataset_adapted)
    return train_dataset_adapted
