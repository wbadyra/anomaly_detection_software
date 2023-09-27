import pandas as pd
import math
import json
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gspread
from config import *
import numpy as np
import csv

def read_data(filename=None): 
    if filename is None:
        filename = '/home/wasyl/magisterka/fault_detection/processed_data/datasets/Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv'
    
    df = pd.read_csv(filename)
    return df
    
def get_true_labels(source_data, error_col):
    """Function which retrieves true data labels.

    Args:
        source_data (numpy.array) - input array
        error_col (list) - list of colums with data labels

    Returns:
        (numpy.array) - array of data labels
    """
    true_labels_temp = np.zeros(shape=(source_data.shape[0]), dtype='int64')
    for col in error_col:
        vals = source_data.loc[:, f'error_{col}'].to_numpy(dtype='int64')
        true_labels_temp = np.add(true_labels_temp, vals)

    return np.where(true_labels_temp==0, 0, 1)


def to_0_1_range(arr, value_as_one):
    """Function which maps proper classes to given values.

    Args:
        arr (numpy.array) - input array
        value_as_one (Int) - value representing class with label 1

    Returns:
        res (list) - list of properly mapped classes
    """
    res = []
    for a in arr:
        if a == value_as_one:
            res.append(1)
        else:
            res.append(0)
    return res

def resume_from_previous_fail():
    """Function which retrieves parameters from last experiment.

    Returns:
        last_params (list) - list of lastly used parameters
    """
    with open("last_run_config.json", "r") as infile:
        last_params = json.load(infile)
    
    return last_params

def found_last_used_param(current_value, param_type):
    """Function which checks, if currently used parameter was used in last experiment.
    
    Args:
        current_values (String) - current parameter value
        param_type (String) - current parameter type
    
    Returns:
        (Bool) - information if proper parameter value was found
    """
    last_params = resume_from_previous_fail()
    if current_value!=last_params[param_type]:
        return False
    return True  
    
def save_last_configuration(params):
    """Saves parameters of last experiment's configuration.

    Args:
        params (list) - list of used parameters
    """
    with open("last_run_config.json", "w") as outfile:
        json.dump(params, outfile)

def metrics(y_true, y_pred, visualization=True):
    """Returns all 5 performance metrics, used for model performance evaluation.

    Args:
        y_true (numpy.array or pandas.Series): True values of target parameter
        y_pred (numpy.array or pandas.Series): Predicted by model values of target parameter
    Returns:
        5 performance metrics (float): accuracy, sensitivity, specificity, F1_score, AUC
    """
    # accuracy, sensitivity, specificity, F1-score
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = report['accuracy']
    sensitivity = report['0']['recall']
    specificity = report['1']['recall']
    F1_score = report['weighted avg']['f1-score']
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    if AUC=='NaN' or AUC=='nan':
        AUC = 'error'
        print("Value of AUC was not calculated properly!")
    if visualization:
        # confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()
    return accuracy, sensitivity, specificity, F1_score, AUC


def save_data(model_name, dimension_reduction, description, accuracy, sensitivity, specificity, F1_score, AUC,
              sheet_name=None, params=None):
    """Save model performance metrics in given directory

    Args:
        model_name (String): Name od model.
        dimension_reduction (String): Used dimension reduction method.
        description (String): Additional description.
        accuracy (float): Accuracy metric value.
        sensitivity (float): Sensitivity metric value.
        specificity (float): Specificity metric value.
        F1_score (float): F1 score metric value.
        AUC (float): Area under curve metric value.
        sheet_name (String): Name of data sheet inside spreadsheet to save results.
        params (list): additional values to be saved.
    """
    save_last_configuration(params)
    
    if params is not None:
        filename = params["filename"]
        error_type = params["errors_types"]
        error_col =  params["columns_with_errors"]
        dataset_name = dir_names_dict[params["directories"]]
        
    fileout = "example_results_filename.csv"
        
    with open(fileout, 'a') as f:
    # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow([model_name, dimension_reduction, description, accuracy, sensitivity, specificity, F1_score, str(AUC), filename, dataset_name, error_type, ''.join(str(e) for e in error_col)])

        
        
        