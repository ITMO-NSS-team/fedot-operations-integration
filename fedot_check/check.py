import os
import datetime
import pandas as pd
import numpy as np

from fedot.core.data.data_split import train_test_data_setup

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.validation.split import tabular_cv_generator
from sklearn.metrics import f1_score, mean_absolute_error

from fedot_check.data import parse_dataframe
from fedot_check.paths import classification_data_path, regression_data_path


def initial_pipeline_with_desired_root(root_model: str):
    node_encoding = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoding])
    node_final = SecondaryNode(root_model, nodes_from=[node_scaling])
    pipeline = Pipeline(node_final)
    return pipeline


class InitialAssumptionsChecker:

    def __init__(self, model_to_check):
        self.regression_model = model_to_check['regression']
        self.classification_model = model_to_check['classification']

        self.classification_base = 'rf'
        self.regression_base = 'rfr'

        self.tuning_repeats = 20
        self.simple_repeats = 5

        # Get all datasets for both tasks
        self.classification_files = os.listdir(classification_data_path())
        self.classification_files.sort()
        self.regression_files = os.listdir(regression_data_path())
        self.regression_files.sort()

    def simple_test(self):
        """ Launch test without tuning for classification and regression datasets """
        simple_validation(task_name='classification', files=self.classification_files,
                          repeats=self.simple_repeats, base_model=self.classification_base,
                          advanced_model=self.classification_model)

        simple_validation(task_name='regression', files=self.regression_files,
                          repeats=self.simple_repeats, base_model=self.regression_base,
                          advanced_model=self.regression_base)

    def advanced_test(self, tuning_iterations: int):
        """ Launch test with tuning each initial assumption """
        for dataset in self.classification_files:
            dataset_name = dataset.split('.csv')[0]
            print(f'Process dataset {dataset}')

            for i in range(self.simple_repeats):
                # Load file
                df = pd.read_csv(os.path.join(classification_data_path(), dataset))
                input_data = parse_dataframe(dataset_name, df)

                # Cross-validation
                metrics_for_folds = []
                time_for_folds = []
                for train_data, test_data in tabular_cv_generator(input_data, folds=4):
                    # Generate pipelines
                    pass


def calculate_metric(task_name, test_target, forecasted_values):
    if task_name == 'classification':
        metric = f1_score(test_target, forecasted_values, average='weighted')
        metric_name = 'F1 metric test score'
    else:
        metric = mean_absolute_error(test_target, forecasted_values)
        metric_name = 'MAE metric test score'

    return metric, metric_name


def simple_validation(task_name: str, files: list, repeats: int, base_model: str, advanced_model: str):
    """ Launch simplified """
    path_by_task = {'classification': classification_data_path(),
                    'regression': regression_data_path()}
    file_by_task = {'classification': 'classification_simple_report.csv',
                    'regression': 'regression_simple_report.csv'}
    output_mode_by_task = {'classification': 'labels',
                           'regression': 'default'}

    output_mode = output_mode_by_task.get(task_name)
    path_to_files = path_by_task.get(task_name)

    results = []
    for dataset in files:
        dataset_name = dataset.split('.csv')[0]
        print(f'Process dataset {dataset}')

        for i in range(repeats):
            # Load file
            df = pd.read_csv(os.path.join(path_to_files, dataset))
            input_data = parse_dataframe(dataset_name, df)

            for model_name in [base_model, advanced_model]:
                pipeline = initial_pipeline_with_desired_root(model_name)
                train_data, test_data = train_test_data_setup(input_data)

                starting_time = datetime.datetime.now()
                pipeline.fit(train_data)
                prediction = pipeline.predict(test_data, output_mode=output_mode)
                forecasted_values = np.array(prediction.predict)

                spend_time = datetime.datetime.now() - starting_time
                fit_minutes = spend_time.total_seconds() / 60

                metric, metric_name = calculate_metric(task_name, test_data.target, forecasted_values)
                results.append([dataset_name, i, metric, fit_minutes, model_name])

                print(f'Launch {i}. Model {model_name}. Fit time in minutes {fit_minutes}')

    report = pd.DataFrame(results, columns=['Dataset', 'Launch id', metric_name,
                                            'Fit minutes', 'Model name'])
    report.to_csv(file_by_task.get(task_name), index=False)
