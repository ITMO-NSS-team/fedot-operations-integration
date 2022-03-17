import datetime

import pandas as pd
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

CLASSIFICATIONS_TABLES = ['volkert', 'cnae-9', 'vehicle', 'Amazon_employee_access']


def parse_dataframe(dataset_name: str, df: pd.DataFrame):
    """ Parse dataframe with features and target

    :param dataset_name: name of tabular data
    :param df: table with features and target columns
    """
    if dataset_name == 'volkert':
        features_cols = np.array(df.columns[1:])
        features = np.array(df[features_cols])
        target = np.array(df['class'])
    elif dataset_name == 'cnae-9':
        features_cols = np.array(df.columns[:-1])
        features = np.array(df[features_cols])
        target = np.array(df['Class'])
    elif dataset_name == 'Amazon_employee_access':
        features_cols = np.array(df.columns[:-1])
        features = np.array(df[features_cols])
        target = np.array(df['target'])
    elif dataset_name == 'vehicle':
        features_cols = np.array(df.columns[:-1])
        features = np.array(df[features_cols])
        target = np.array(df['Class'])
    else:
        raise ValueError(f'Dataset with name {dataset_name} does not exist')

    if dataset_name in CLASSIFICATIONS_TABLES:
        input_data = prepare_input_data(features, target, task_name='classification')
    else:
        input_data = prepare_input_data(features, target, task_name='regression')

    return input_data


def prepare_input_data(features, target, task_name: str):
    """ Create input data dataclass for FEDOT """
    if task_name == 'regression':
        task = Task(TaskTypesEnum.regression)
    else:
        task = Task(TaskTypesEnum.classification)
    input_data = InputData(idx=np.arange(0, len(target)), features=features,
                           target=target, task=task,
                           data_type=DataTypesEnum.table)

    return input_data
