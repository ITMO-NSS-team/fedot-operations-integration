import datetime

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline


class ComposingChecker:
    """ Launch experiments with composing stage """

    def __init__(self, model_to_check):
        self.regression_model = model_to_check['regression']
        self.classification_model = model_to_check['classification']

    def legacy_launch(self):
        starting_time = datetime.datetime.now()
        init_pipeline = Pipeline(PrimaryNode('lgbm'))
        composer_params = {'history_folder': 'custom_history_folder',
                           'initial_assumption': init_pipeline}
        auto_model = Fedot(problem=problem, seed=42, composer_params=composer_params,
                           timeout=5, verbose_level=1)
        auto_model.fit(train_input)
        prediction = auto_model.predict(test_input)

        spend_time = datetime.datetime.now() - starting_time
        composing_spend_time = auto_model.api_composer.timer.composing_spend_time
        tuning_spend_time = auto_model.api_composer.timer.tuning_spend_time

        all_minutes = spend_time.total_seconds() / 60
        composing_minutes = composing_spend_time.total_seconds() / 60
        tuning_minutes = tuning_spend_time.total_seconds() / 60
        print(f'Spend time for all calculations: {all_minutes:.2f}')
        print(f'Spend time for composing: {composing_minutes:.2f}')
        print(f'Spend time for tuning: {tuning_minutes:.2f}')
