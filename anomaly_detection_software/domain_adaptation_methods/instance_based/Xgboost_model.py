import json

from sklearn.metrics import classification_report, roc_curve, auc
from xgboost import XGBClassifier

def XGBoost_model():
    """Returns XGBoost classifier model. All hyperparameters values are read from json file.

    Returns:
        XGBClassifier model
    """
    model = XGBClassifier(base_score=0.5, learning_rate=0.5)
    return model


def metrics(y_true, y_pred):
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
    # f1 score is taken as weighted average of classes
    F1_score = report['weighted avg']['f1-score']
    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    return accuracy, sensitivity, specificity, F1_score, AUC