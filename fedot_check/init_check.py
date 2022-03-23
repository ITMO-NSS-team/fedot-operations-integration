import os
import datetime
import pandas as pd
import numpy as np

from fedot.core.data.data_split import train_test_data_setup

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.validation.split import tabular_cv_generator
from sklearn.metrics import mean_absolute_error, roc_auc_score

from fedot_check.constants import OUTPUT_MODE_BY_TASK, PATH_BY_TASK, FILE_BY_TASK, ITERATIONS_FOR_TUNING
from fedot_check.data import parse_dataframe, load_csv_dataframe
from fedot_check.metrics import calculate_metric
from fedot_check.paths import classification_data_path, regression_data_path


class InitialAssumptionsChecker:

    def __init__(self, models_to_check):
        self.regression_model = models_to_check['regression']
        self.classification_model = models_to_check['classification']

        self.classification_base = 'rf'
        self.regression_base = 'rfr'

        self.tuning_repeats = 20
        self.simple_repeats = 5

        # Get all datasets for both tasks
        self.classification_files = os.listdir(classification_data_path())
        self.classification_files.sort()
        self.regression_files = os.listdir(regression_data_path())
        self.regression_files.sort()

    def simple_assumption_validation(self):
        """ Launch test without tuning for classification and regression datasets """
        simple_validation(task_name='classification', files=self.classification_files,
                          repeats=self.simple_repeats, base_model=self.classification_base,
                          advanced_model=self.classification_model)

        simple_validation(task_name='regression', files=self.regression_files,
                          repeats=self.simple_repeats, base_model=self.regression_base,
                          advanced_model=self.regression_model)

    def advanced_assumption_validation(self, tuning_iterations: int):
        """ Launch test with tuning each initial assumption """
        advanced_validation(task_name='classification', files=self.classification_files,
                            repeats=self.tuning_repeats, base_model=self.classification_base,
                            advanced_model=self.classification_model)

        advanced_validation(task_name='regression', files=self.regression_files,
                            repeats=self.tuning_repeats, base_model=self.regression_base,
                            advanced_model=self.regression_model)


def initial_pipeline_with_desired_root(root_model: str):
    node_encoding = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoding])
    node_final = SecondaryNode(root_model, nodes_from=[node_scaling])
    pipeline = Pipeline(node_final)
    return pipeline


def simple_validation(task_name: str, files: list, repeats: int, base_model: str, advanced_model: str):
    """ Launch simplified validation. Fit initial pipelines and compare fit time and metrics

    :param task_name: name of task
    :param files: names of csv files
    :param repeats: number of launches
    :param base_model: name of baseline model for root node
    :param advanced_model: name of new model for root node
    """
    output_mode = OUTPUT_MODE_BY_TASK.get(task_name)
    path_to_files = PATH_BY_TASK.get(task_name)

    results = []
    for dataset in files:
        dataset_name = dataset.split('.csv')[0]
        print(f'Process dataset {dataset}')

        for i in range(repeats):
            df = load_csv_dataframe(path_to_files, dataset)
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
    report.to_csv(FILE_BY_TASK.get(task_name), index=False)


def advanced_validation(task_name: str, files: list, repeats: int, base_model: str, advanced_model: str):
    """
    Validate initial assumption with tuning (so hyperparameters space investigated)

    :param task_name: name of task to solve
    :param files: names of csv files
    :param repeats: number of launches
    :param base_model: name of baseline model for root node
    :param advanced_model: name of new model for root node
    """
    output_mode = OUTPUT_MODE_BY_TASK.get(task_name)
    path_to_files = PATH_BY_TASK.get(task_name)

    results = []
    for dataset in files:
        dataset_name = dataset.split('.csv')[0]
        print(f'Process dataset {dataset}')

        for i in range(repeats):
            # Load file
            df = load_csv_dataframe(path_to_files, dataset)
            input_data = parse_dataframe(dataset_name, df)

            for model_name in [base_model, advanced_model]:
                pipeline = initial_pipeline_with_desired_root(model_name)

                # Cross-validation
                metrics_for_folds = []
                time_for_folds = []
                for train_data, test_data in tabular_cv_generator(input_data, folds=4):
                    starting_time = datetime.datetime.now()
                    if task_name == 'classification':
                        pipeline = pipeline.fine_tune_all_nodes(input_data=train_data,
                                                                loss_function=roc_auc_score,
                                                                loss_params={'multi_class': 'ovr',
                                                                             'average': 'weighted'},
                                                                iterations=ITERATIONS_FOR_TUNING,
                                                                timeout=25)
                    else:
                        # Tuning for regression
                        pipeline = pipeline.fine_tune_all_nodes(input_data=train_data,
                                                                loss_function=mean_absolute_error,
                                                                iterations=ITERATIONS_FOR_TUNING,
                                                                timeout=25)
                    prediction = pipeline.predict(test_data, output_mode=output_mode)
                    forecasted_values = np.array(prediction.predict)

                    spend_time = datetime.datetime.now() - starting_time
                    fit_minutes = spend_time.total_seconds() / 60

                    metric, metric_name = calculate_metric(task_name, test_data.target, forecasted_values)
                    metrics_for_folds.append(metric)
                    time_for_folds.append(fit_minutes)
                # Calculate mean values
                metrics_for_folds = np.array(metrics_for_folds)
                time_for_folds = np.array(time_for_folds)

                mean_metric = float(np.mean(metrics_for_folds))
                mean_fit_time = float(np.mean(time_for_folds))

                results.append([dataset_name, i, mean_metric, mean_fit_time, model_name])
                print(f'Launch {i}. Model {model_name}. Fit time in minutes {mean_fit_time}')

    report = pd.DataFrame(results, columns=['Dataset', 'Launch id', metric_name,
                                            'Fit minutes', 'Model name'])
    report.to_csv(FILE_BY_TASK.get(task_name), index=False)
