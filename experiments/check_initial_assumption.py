from fedot_check.check import InitialAssumptionsChecker
from fedot_check.paths import classification_data_path


def check_initial_assumption(model_to_check: dict):
    """
    Generates initial assumptions with initial models and with new models in root nodes.
    After that on all datasets it measures running time and metrics on validation sample
    """
    checker = InitialAssumptionsChecker(model_to_check)

    # Launch validation for model with default hyperparameters
    checker.simple_test()

    # Vary hyperparameters for model
    checker.advanced_test(tuning_iterations=10)


if __name__ == '__main__':
    check_initial_assumption(model_to_check={'regression': 'lgbmreg',
                                             'classification': 'lgbm'})
