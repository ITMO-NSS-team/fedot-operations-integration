from fedot_check.check import InitialAssumptionsChecker
from fedot_check.visualization import display_initial_assumptions_comparison


def check_initial_assumption(model_to_check: dict):
    """
    Generates initial assumptions with initial models and with new models in root nodes.
    After that on all datasets it measures running time and metrics on validation sample
    """
    checker = InitialAssumptionsChecker(model_to_check)

    # Launch validation for model with default hyperparameters
    # checker.simple_assumption_validation()

    # Vary hyperparameters for model
    checker.advanced_assumption_validation(tuning_iterations=10)


def plot_graphs():
    display_initial_assumptions_comparison('classification_simple_report.csv')
    display_initial_assumptions_comparison('regression_simple_report.csv')

    display_initial_assumptions_comparison('classification_advanced_report.csv')
    display_initial_assumptions_comparison('regression_advanced_report.csv')


if __name__ == '__main__':
    check_initial_assumption(model_to_check={'regression': 'lgbmreg',
                                             'classification': 'lgbm'})
    plot_graphs()
