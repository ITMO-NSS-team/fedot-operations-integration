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
