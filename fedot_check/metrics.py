import numpy as np

from sklearn.metrics import f1_score


def calculate_metric(task_name, test_target, forecasted_values):
    if task_name == 'classification':
        metric = f1_score(test_target, forecasted_values, average='weighted')
        metric_name = 'F1 metric test score'
    else:
        metric = smape_metric(test_target, forecasted_values)
        metric_name = 'SMAPE metric test score'

    return metric, metric_name


def smape_metric(y_true: np.array, y_pred: np.array) -> float:
    """ Symmetric mean absolute percentage error """

    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = numerator / denominator
    result[np.isnan(result)] = 0.0
    return float(np.mean(100 * result))
