import pandas as pd
import math
import json
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gspread
from gspread import utils
from oauth2client.service_account import ServiceAccountCredentials
from config import *
import numpy as np
import csv

def read_data(filename=None):
    if filename is None:
        filename = '/home/wasyl/magisterka/fault_detection/processed_data/datasets/Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv'
    
    # file = '/home/wasyl/magisterka/fault_detection/processed_data/datasets/Kia_soul_after_da/Kia_soul_after_da_combined.csv'
    # file = '../datasets/Dataset/A/All_1.csv'
    # file = '../datasets/carOBD-master/obdiidata/drive1.csv'
    # file = '../datasets/camera_lidar/20180810_150607/bus/data_processed_v2.csv'
    # file = '../datasets/archive/exp1_14drivers_14cars_dailyRoutes.csv'
    df = pd.read_csv(filename)
    return df


def round_half_away_from_zero(n):
    if n >= 0.0:
        return math.ceil(n)
    else:
        return math.floor(n)
    
def get_true_labels(source_data, error_col):
    true_labels_temp = np.zeros(shape=(source_data.shape[0]), dtype='int64')
    for col in error_col:
        vals = source_data.loc[:, f'error_{col}'].to_numpy(dtype='int64')
        true_labels_temp = np.add(true_labels_temp, vals)

    return np.where(true_labels_temp==0, 0, 1)


def to_0_1_range(arr, value_as_one):
    res = []
    for a in arr:
        if a == value_as_one:
            res.append(1)
        else:
            res.append(0)
    return res

def resume_from_previous_fail():
    with open("last_run_config.json", "r") as infile:
        last_params = json.load(infile)
    
    return last_params

def found_last_used_param(current_value, param_type):
    last_params = resume_from_previous_fail()
    if current_value!=last_params[param_type]:
        return False
    return True  
    
def save_last_configuration(params):
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
    #validate if y_true and y_pred both classes
    vals, count = np.unique(y_true, return_counts=True)
    if len(count) == 1:
        print("There is no errors in data!!!")
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = report['accuracy']
    sensitivity = report['0']['recall']
    specificity = report['1']['recall']
    # try:
    #     specificity = report['1']['recall']
    # except KeyError:
    #     specificity = 0.0
    # f1 score is taken as weighted average of classes
    F1_score = report['weighted avg']['f1-score']
    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    if AUC=='NaN' or AUC=='nan':
        AUC = 'error'
        print("Value of AUC was not calculated properly!")
    if visualization:
        # confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()
    return accuracy, sensitivity, specificity, F1_score, AUC


def create_spreadsheet(client):
    sheet = client.create("fault_detection_results")
    sheet.share('p.frackowski12191@gmail.com', perm_type='user', role='writer')


def save_data(model_name, dimension_reduction, description, accuracy, sensitivity, specificity, F1_score, AUC,
              sheet_name=None, params=None):
    """Save model performance metrics to google spreadsheet.

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
    """
    save_last_configuration(params)
    # Create new spreadsheet only once
    # create_spreadsheet(client)

    # Open the spreadsheet
    gc = gspread.service_account()
    
    last_col = ':H'
    if params is not None:
        filename = params["filename"]
        error_type = params["errors_types"]
        error_col =  params["columns_with_errors"]
        dataset_name = dir_names_dict[params["directories"]]
        
        last_col = ':L'
    
    fileout = "results_exp1.csv"
    if len(error_col) == 1:
        fileout = "results_1error_exp1.csv"
    elif len(error_col) == 2:
        fileout = "results_2errors_exp1_same_time.csv"
    elif len(error_col) == 3:
        fileout = "results_3errors_exp1_same_time.csv"
        
    with open(fileout, 'a') as f:
    # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow([model_name, dimension_reduction, description, accuracy, sensitivity, specificity, F1_score, str(AUC), filename, dataset_name, error_type, ''.join(str(e) for e in error_col)])

    # OVERRIDE
    # sheet_name = ['results5']

    # if sheet_name is None:
    #     sheet_name = [model_name, dimension_reduction, error_type, dataset_name]
    
    # sheet = gc.open("fault_detection_results")
    # for sh in sheet_name:
    #     # time.sleep(15)
    #     # print("Sleeping")
    #     try:
    #         worksheet = sheet.worksheet(title=str(sh))
    #     except:
    #         worksheet = sheet.add_worksheet(title=str(sh), rows=10000, cols=15)

    #     row = len(worksheet.col_values(1)) + 1
    #     data = [{
    #         "range": f'A{str(row)}{last_col}{str(row)}', 
    #         "values": [[model_name, dimension_reduction, description, accuracy, sensitivity, specificity, F1_score, str(AUC), filename, dataset_name, error_type, ''.join(str(e) for e in error_col)]]}]
        
    #     worksheet.batch_update(data)
        
        
        