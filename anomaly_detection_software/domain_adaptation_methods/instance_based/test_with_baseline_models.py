import glob

from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import json
import os

from sklearn.preprocessing import normalize

from transform_features import dataset_adaptation
from sklearn.model_selection import train_test_split
# from baseline_models.utils import save_model
from sklearn.utils import shuffle
from Xgboost_model import XGBoost_model, metrics

# def prepare_datasets(dir, list_of_datasets):
#     # remember to normalize
#     datasets = []
#     for dataset in list_of_datasets:
#         with open(f"{dir}/{dataset}.csv", "r") as fileout:
#             pass



PARAMS = [0.4, 0.6, 0.8, 1.0]
RESULT_FILE = 'filename.txt'
MODELS_LIST = ["XGBoost"]
ADAPTATION_METHODS_LIST = ["Balanced_Weighting", "WANN", "KMM", "KLIEP", "TrAdaBoost"]

def adaptation_models(train_dataset, validation_dataset, test_dataset, algorithm, model, dataset_names, experiment,
                      params=None):
    """Prepares model and trains it on adapted data.

    Args:
        train_dataset (pandas.DataFrame): Training data for model.
        validation_dataset (pandas.DataFrame): Validation data for model.
        test_dataset (pandas.DataFrame): Test data for model.
        algorithm (String): Selection of domain adaptation method.
        model (Object): Machine learning model.
        dataset_names (List): List of dataset names which are included in experiment.
        experiment (String): Experiment name, saves results in proper destination.
        params (List): Optional, if given data is adapted for each value in list.

    Returns:
        model (Object): Model with learned weights.
        x_test (pandas.Dataframe): Features of test data
        y_test (List): Labels of test data
    """

    # divide validation and test data to features and labels
    x_val = validation_dataset.drop(['target'], axis=1).values
    y_val = validation_dataset['target'].values
    x_test = test_dataset.drop(['target'], axis=1).values
    y_test = test_dataset['target'].values

    # calculate train_dataset_adapted with given DA algorithm and divide into features and labels
    train_dataset_adapted = dataset_adaptation(train_dataset, algorithm, dataset_names, experiment, params=params)

    x_train = train_dataset_adapted.drop(['target'], axis=1).values
    y_train = train_dataset_adapted['target'].values

    # shuffle data
    seed = json.load(open('/home/wasyl/projekt_badawczy/cbc_covid/baseline_models/hyperparameters.json'))['random_state']
    x_train, y_train = shuffle(x_train, y_train, random_state=seed)

    # given model_name properly implement early stopping and fit data to the model
    # if model_name == "tensorflow":
    #     callback = EarlyStopping(monitor='accuracy', patience=10)
    #     model.fit(x_train, y_train, validation_split=0.1, callbacks=[callback])
    # elif model_name == "TabNet":
    #     model.fit(X_train=x_train, y_train=y_train,
    #               eval_set=[(x_val, y_val)],
    #               eval_name=["val"], eval_metric=['accuracy'], max_epochs=500)
    # elif model_name == "ANN":
    #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    #     model.fit(x_train, y_train,
    #               validation_data=(x_val, y_val), epochs=500, callbacks=[es])
    # elif model_name == ["CatBoost", "XGBoost"]:
    #     model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=10)
    # else:
    model.fit(x_train, y_train)
    return model, x_test, y_test


def load_datasets(train_dir, test_dir, train_names, test_names):
    """Loads both train and test data from datasets with given name at given directory.

    Args:
        train_dir (String): Location af train datasets.
        test_dir (String): Location af test datasets.
        train_names (List): List of dataset names used in training model
        test_names (List): List of dataset names used in testing model

    Returns:
        train_data (List): List of pandas.Dataframes with train data from each country.
        validation_data (List): List of pandas.Dataframes with validation data from each country.
        test_data (List): List of pandas.Dataframes with test data from each country.
        names (List): List of names of datasets used in training process
    """

    train_data = []
    test_data = []
    validation_data = []
    names = []
    for filename in os.listdir(train_dir):
        train_f = os.path.join(train_dir, filename)
        test_f = os.path.join(test_dir, filename)
        # checking if it is a file
        if os.path.isfile(train_f) and filename[-3:] == 'csv':
            X_train = pd.read_csv(train_f, index_col=0)
            X_test = pd.read_csv(test_f, index_col=0)
            # data normalization
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
            X_train['target'] = X_train['target'].astype(int)
            X_test['target'] = X_test['target'].astype(int)
            for name in test_names:
                if name.lower() in filename.lower():
                    test_data.append(X_test)
            for name in train_names:
                if name.lower() in filename.lower():
                    train_data.append(X_train)
                    names.append(filename)
                    validation_data.append(X_test)

    return train_data, validation_data, test_data, names


def prepare_dataset(train_data, validation_data, test_data, external_validation=False):
    """Prepares datasets, creates single dataframe for validation and test data.

    Args:
        train_data (List): List of pandas.Dataframes with train data from each country.
        validation_data (List): List of pandas.Dataframes with validation data from each country.
        test_data (List): List of pandas.Dataframes with test data from each country.
        external_validation (Boolean): Whether the experiment is an external validation.

    Returns:
        train_data (List): List of pandas.Dataframes with train data from each country with sorted columns.
        validation_dataset_split (pandas.Dataframe): Concatenated validation data.
        test_dataset_split (pandas.Dataframe): Concatenated test data.
    """

    for i in range(len(train_data)):
        train_data[i] = train_data[i].reindex(sorted(train_data[i].columns), axis=1)

    frames_list = []
    for i in range(len(test_data)):
        test_data[i] = test_data[i].reindex(sorted(test_data[i].columns), axis=1)
        frames_list.append(test_data[i])
    test_dataset = pd.concat(frames_list, ignore_index=True)

    frames_list = []
    if not external_validation:
        # if not external_validation we must divide our test_dataset to validation and test data
        seed = json.load(open('/home/wasyl/projekt_badawczy/cbc_covid/baseline_models/hyperparameters.json'))[
            'random_state']
        validation_dataset_split, test_dataset_split = train_test_split(test_dataset, train_size=0.5, random_state=seed)
    else:
        for i in range(len(validation_data)):
            validation_data[i] = validation_data[i].reindex(sorted(validation_data[i].columns), axis=1)
            frames_list.append(validation_data[i])
        validation_dataset_split = pd.concat(frames_list, ignore_index=True)
        test_dataset_split = test_dataset
    return train_data, validation_dataset_split, test_dataset_split


def train_model(train_dataset, validation_dataset, test_dataset, model_name, algorithm, train_names, test_names,
                experiment):
    """Trains given model with adapted data by given method.

    Args:
        train_dataset (pandas.DataFrame): Training data for model.
        validation_dataset (pandas.DataFrame): Validation data for model.
        test_dataset (pandas.DataFrame): Test data for model.
        model_name (String): Name of the model.
        algorithm (String): Selection of domain adaptation method.
        train_names (List): List of dataset names which are included in training.
        test_names (List): List of dataset names which are included in testing.
        experiment (String): Experiment name, saves results in proper destination.
    """

    # whether there are globally declared PARAMS for DA methods
    if PARAMS is not None:
        for param in PARAMS:
            model, x_test, y_test = adaptation_models(train_dataset, validation_dataset, test_dataset, algorithm,
                                                      XGBoost_model(), train_names,
                                                      experiment, params=param)
            Y_pred = model.predict(x_test)
            Y_pred = (Y_pred > 0.5)
            accuracy_a, sensitivity_a, specificity_a, F1_score_a, AUC_a = metrics(y_test, Y_pred)

            # save results in txt file and save the model
            train_dataset_names = ""
            for name in train_names:
                train_dataset_names += name + "_"
            test_dataset_names = ""
            for name in test_names:
                test_dataset_names += name + "_"
            filename = f"{train_dataset_names}train_{test_dataset_names}test_param_{param}"
            # save_model(model, model_name, filename, F1_score_a, experiment=experiment, algorithm=algorithm, with_da=True)
            with open(RESULT_FILE, 'a') as result_file:
                result_file.write(
                    f"Results for {model_name}, method: {algorithm} with param: {param}, source file is {train_names}, "
                    f"target one is {test_names}. With adaptation:\n accuracy: {accuracy_a},\n sensivity: "
                    f"{sensitivity_a},\nspecificity: {specificity_a},\n F1 score: {F1_score_a},\n AUC: {AUC_a}.\n\n\n")
    else:
        model, x_test, y_test = adaptation_models(train_dataset, validation_dataset, test_dataset, algorithm,
                                  XGBoost_model(), train_names, experiment)
        Y_pred = model.predict(x_test)
        Y_pred = (Y_pred > 0.5)
        accuracy_a, sensitivity_a, specificity_a, F1_score_a, AUC_a = metrics(y_test, Y_pred)

        # save results in txt file and save the model
        train_dataset_names = ""
        for name in train_names:
            train_dataset_names += name + "_"
        test_dataset_names = ""
        for name in test_names:
            test_dataset_names += name + "_"
        filename = f"{train_dataset_names}train_{test_dataset_names}test"
        # save_model(model, model_name, filename, F1_score_a, experiment=experiment, algorithm=algorithm, with_da=True)

        with open(RESULT_FILE, 'a') as result_file:
            result_file.write(
                    f"Results for {model_name}, method: {algorithm}, source file is {train_names}, target one is "
                    f"{test_names}. With adaptation:\n accuracy: {accuracy_a},\n sensivity: {sensitivity_a},\n "
                    f"specificity: {specificity_a},\n F1 score: {F1_score_a},\n AUC: {AUC_a}.\n\n\n")


if __name__ == '__main__':

    # here i only need this method
    # dataset_adaptation
    # as the only thing i do is to merge all files and do adaptation, no dividing into test and train
    dataset_name = "Dataset"
    directory = f"/home/wasyl/magisterka/fault_detection/processed_data/datasets/{dataset_name}/"

    os.chdir(directory)
    extension = 'csv'
    # Time,Engine Coolant Temperature [C],Intake Manifold Pressure [kPa],Engine RPM [RPM],Vehicle Speed [km/h],Throttle Position [%],Accelerator Pedal Position [%],Intake Air Temperature [C],MAF [g/s], ,Ambient Air Temperature [C]
    # Time,Engine Coolant Temperature [C],Intake Manifold Pressure [kPa],Engine RPM [RPM],Vehicle Speed [km/h],Throttle Position [%],Accelerator Pedal Position [%],Steering Wheel Angle [deg]
    # Time,Engine Coolant Temperature [C],Intake Manifold Pressure [kPa],Engine RPM [RPM],Vehicle Speed [km/h],Throttle Position [%],Accelerator Pedal Position [%],Steering Wheel Angle [deg]
    # Time,Engine Coolant Temperature [C],Intake Manifold Pressure [kPa],Engine RPM [RPM],Vehicle Speed [km/h],Throttle Position [%],Accelerator Pedal Position [%],Intake Air Temperature [C]
    # all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    all_filenames = ["/home/wasyl/magisterka/fault_detection/processed_data/datasets/Kia_soul/driver_17.csv",
                     "/home/wasyl/magisterka/fault_detection/processed_data/datasets/Dataset/B_All_6.csv",
                     "/home/wasyl/magisterka/fault_detection/processed_data/datasets/carOBD-master/drive10.csv",
                     "/home/wasyl/magisterka/fault_detection/processed_data/datasets/OBD-II-Dataset/2017-07-27_Seat_Leon_KA_KA_2_Normal.csv"]
    # combine all files in the list
    all_dataset_files = []
    for file in all_filenames:
        dataframe = pd.read_csv(file)
        dataframe = dataframe.dropna()
        # error_list = dataframe.columns.tolist()
        # dataframe_with_errors = insert_fault_erratic(dataframe, error_list[1:], 0.04, 0.1, 5)
        all_dataset_files.append(pd.read_csv(file))

    # print(dataframe_with_errors["error_Engine Coolant Temperature [C]"], dataframe_with_errors["Engine Coolant Temperature [C]"])
    train_datasets = []
    for dataframe in all_dataset_files:
        for col in ["Intake Air Temperature [C]", "MAF [g/s]", "Ambient Air Temperature [C]", "Steering Wheel Angle [deg]"]:
            if col in list(dataframe.columns):
                dataframe.drop(col, axis=1, inplace=True)
        # dataframe = insert_fault_erratic(dataframe, ['Engine RPM [RPM]'], 0.2, 0.4, 10)
        # dataframe.rename(columns={'error_Engine RPM [RPM]': 'target'}, inplace=True)
        # dataframe = dataframe.drop(['Engine RPM [RPM]_health'], axis=1)
        # print(dataframe.columns)
        dataframe['target'] = np.zeros(shape=len(dataframe))
        train_datasets.append(dataframe)

    train_dataset_adapted = dataset_adaptation(train_datasets, "WANN", all_filenames, "test")
    print(train_dataset_adapted.columns)
    train_dataset_adapted.drop(["target"], axis=1)
    train_dataset_adapted.to_csv("/home/wasyl/magisterka/fault_detection/processed_data/datasets/mixed/test1.csv", index=False)


    # dirs = json.load(open('/home/wasyl/projekt_badawczy/cbc_covid/baseline_models/directories.json'))
    # train_dir = dirs['train_dir']
    # test_dir = dirs['test_dir']
    #
    # TEST_DATASETS = ["uck"]
    # TRAIN_DATASETS = ["Italy-4", "uck"]
    # EXPERIMENT = "exp2"
    #
    # train_data, validation_data, test_data, filenames = load_datasets(train_dir, test_dir,
    #                                                                   TRAIN_DATASETS, TEST_DATASETS)
    # train_dataset, validation_dataset, test_dataset = prepare_dataset(train_data, validation_data, test_data)
    #
    # for model_name in MODELS_LIST:
    #     for adaptation_method in ADAPTATION_METHODS_LIST:
    #         RESULT_FILE = f'results/{model_name}/{adaptation_method}/train_uck_zenodo_test_uck.txt'
    #         print(f"Current model: {model_name}, method: {adaptation_method}\n")
    #         train_model(train_dataset, validation_dataset, test_dataset, model_name, adaptation_method,
    #                     TRAIN_DATASETS, TEST_DATASETS, EXPERIMENT)
