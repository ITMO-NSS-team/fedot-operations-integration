from fedot_check.composing_check import AutoMLChecker
from fedot_check.visualization import automl_timeouts_comparison


def check_composing_perform_correctly(models_to_check):
    """
    Launch experiments with composing and tuning stages.
    Start with initial assumption with desired model.
    """
    auto_checker = AutoMLChecker(models_to_check)
    # Single process mode
    auto_checker.composing_tuning_validation(n_jobs=1)

    # Parallel mode
    auto_checker.composing_tuning_validation(n_jobs=-1)


if __name__ == '__main__':
    # check_composing_perform_correctly(models_to_check={'regression': 'lgbmreg',
    #                                                    'classification': 'lgbm'})
    # All csv files saved in current folder
    automl_timeouts_comparison('')
