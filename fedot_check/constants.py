from fedot_check.paths import classification_data_path, regression_data_path


PATH_BY_TASK = {'classification': classification_data_path(),
                'regression': regression_data_path()}
SIMPLE_FILE_BY_TASK = {'classification': 'classification_simple_report.csv',
                       'regression': 'regression_simple_report.csv'}
ADVANCED_FILE_BY_TASK = {'classification': 'classification_advanced_report.csv',
                         'regression': 'regression_advanced_report.csv'}
OUTPUT_MODE_BY_TASK = {'classification': 'labels',
                       'regression': 'default'}
ITERATIONS_FOR_TUNING = 15
