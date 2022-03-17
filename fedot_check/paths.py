import os
from pathlib import Path


def get_project_path() -> str:
    return Path(__file__).parent.parent


def classification_data_path() -> str:
    return os.path.join(get_project_path(), 'data', 'classification')


def regression_data_path() -> str:
    return os.path.join(get_project_path(), 'data', 'regression')
