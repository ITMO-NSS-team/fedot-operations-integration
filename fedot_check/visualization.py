import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def display_initial_assumptions_comparison(path_file: str):
    df = pd.read_csv(path_file)
    with sns.axes_style("darkgrid"):
        sns.catplot(x='Model name', y='Fit minutes',
                    hue='Model name', col='Dataset',
                    palette='rainbow',
                    data=df, kind="strip", dodge=True,
                    height=4, aspect=.7)
        plt.show()

    # Fined column with metric
    column_names = list(df.columns)
    filtered = [column for column in column_names if 'metric' in column]

    with sns.axes_style("darkgrid"):
        sns.catplot(x='Model name', y=filtered[0],
                    hue='Model name', col='Dataset',
                    palette='coolwarm',
                    data=df, kind="strip", dodge=True,
                    height=4, aspect=.7)
        plt.show()


def automl_timeouts_comparison(path_with_files: str):
    """ Display timeouts for classification and regression results """
    path_with_files = os.path.abspath(path_with_files)
    df_class_single = pd.read_csv(os.path.join(path_with_files, 'classification_automl_single_process.csv'))
    df_class_single = transform_dataset(df_class_single, mode='single process', task='classification')

    df_class_parallel = pd.read_csv(os.path.join(path_with_files, 'classification_automl_parallel.csv'))
    df_class_parallel = transform_dataset(df_class_parallel, mode='parallel', task='classification')

    df_reg_single = pd.read_csv(os.path.join(path_with_files, 'regression_automl_single_process.csv'))
    df_reg_single = transform_dataset(df_reg_single, mode='single process', task='regression')

    df_reg_parallel = pd.read_csv(os.path.join(path_with_files, 'regression_automl_parallel.csv'))
    df_reg_parallel = transform_dataset(df_reg_parallel, mode='parallel', task='regression')

    full_class_dataframe = pd.concat([df_class_single, df_class_parallel])
    full_class_dataframe['Timeout minutes'] = full_class_dataframe['Timeout minutes'].astype(float)

    full_reg_dataframe = pd.concat([df_reg_single, df_reg_parallel])
    full_reg_dataframe['Timeout minutes'] = full_reg_dataframe['Timeout minutes'].astype(float)

    for df, plot_name in zip([full_class_dataframe, full_reg_dataframe], ['classification.png', 'regression.png']):
        with sns.axes_style("darkgrid"):
            sns.catplot(x='Launch id', y='Timeout minutes',
                        col='Mode', row='Dataset',
                        data=df,
                        hue='Stage',
                        height=4, aspect=1.5)
            plt.savefig(os.path.join(path_with_files, plot_name))
            print(f'Save picture {os.path.join(path_with_files, plot_name)}')
            plt.close()


def transform_dataset(dataframe: pd.DataFrame, mode: str, task: str) -> pd.DataFrame:
    # Take timeouts per case: "All", "Composing", "Tuning"
    composing_dataframe = dataframe[['Dataset', 'Launch id', 'Composing minutes']]
    composing_dataframe = composing_dataframe.rename(columns={'Composing minutes': 'Timeout minutes'})
    composing_dataframe['Stage'] = ['composing'] * len(composing_dataframe)

    tuning_dataframe = dataframe[['Dataset', 'Launch id', 'Tuning minutes']]
    tuning_dataframe = tuning_dataframe.rename(columns={'Tuning minutes': 'Timeout minutes'})
    tuning_dataframe['Stage'] = ['tuning'] * len(tuning_dataframe)

    all_dataframe = dataframe[['Dataset', 'Launch id', 'All minutes']]
    all_dataframe = all_dataframe.rename(columns={'All minutes': 'Timeout minutes'})
    all_dataframe['Stage'] = ['All'] * len(tuning_dataframe)

    dataframe = pd.concat([composing_dataframe, tuning_dataframe, all_dataframe])
    dataframe['Mode'] = [mode] * len(dataframe)
    dataframe['Task'] = [task] * len(dataframe)
    return dataframe
