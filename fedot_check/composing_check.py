import datetime
import os
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot_check.constants import PATH_BY_TASK
from fedot_check.data import load_csv_dataframe, parse_dataframe
from fedot_check.paths import classification_data_path, regression_data_path


class AutoMLChecker:
    """ Launch experiments with composing and tuning stages """

    def __init__(self, model_to_check):
        self.regression_model = model_to_check['regression']
        self.classification_model = model_to_check['classification']

        self.timeout_minutes = 5
        self.repeats = 10

        self.classification_files = os.listdir(classification_data_path())
        self.classification_files.sort()
        self.regression_files = os.listdir(regression_data_path())
        self.regression_files.sort()

    def composing_tuning_validation(self, n_jobs: int = 1):
        """ Launch AutoML algorithm with composing and tuning """
        run_composing_validation(task_name='classification', files=self.classification_files,
                                 repeats=self.repeats, operation=self.classification_model,
                                 timeout_minutes=self.timeout_minutes, n_jobs=n_jobs)

        run_composing_validation(task_name='regression', files=self.classification_files,
                                 repeats=self.repeats, operation=self.regression_model,
                                 timeout_minutes=self.timeout_minutes, n_jobs=n_jobs)


def run_composing_validation(task_name: str, files: list, repeats: int, operation: str,
                             timeout_minutes: float, n_jobs: int):
    """ Launch AutoML algorithm with desired parameters. Measure only time - not metrics """
    path_to_files = PATH_BY_TASK.get(task_name)

    results = []
    for dataset in files:
        dataset_name = dataset.split('.csv')[0]
        print(f'Process dataset {dataset}')

        for i in range(repeats):
            df = load_csv_dataframe(path_to_files, dataset)
            input_data = parse_dataframe(dataset_name, df)

            starting_time = datetime.datetime.now()
            init_pipeline = Pipeline(PrimaryNode(operation))

            composer_params = {'initial_assumption': init_pipeline,
                               'available_operations': [operation]}
            auto_model = Fedot(problem=task_name, composer_params=composer_params, timeout=timeout_minutes,
                               verbose_level=0, n_jobs=n_jobs)
            auto_model.fit(input_data)
            spend_time = datetime.datetime.now() - starting_time
            composing_spend_time = auto_model.api_composer.timer.composing_spend_time
            tuning_spend_time = auto_model.api_composer.timer.tuning_spend_time

            all_minutes = spend_time.total_seconds() / 60
            composing_minutes = composing_spend_time.total_seconds() / 60
            tuning_minutes = tuning_spend_time.total_seconds() / 60
            print(f'Spend time for all calculations: {all_minutes:.2f}')
            print(f'Spend time for composing: {composing_minutes:.2f}')
            print(f'Spend time for tuning: {tuning_minutes:.2f}')
            results.append([dataset_name, i, composing_minutes, tuning_minutes, all_minutes])

    column_names = ['Dataset', 'Launch id', 'Composing minutes', 'Tuning minutes', 'All minutes']
    results = pd.DataFrame(results, columns=column_names)

    if n_jobs == -1 or n_jobs > 1:
        file_name = f'{task_name}_automl_parallel.csv'
    else:
        file_name = f'{task_name}_automl_single_process.csv'

    results.to_csv(file_name, index=False)
