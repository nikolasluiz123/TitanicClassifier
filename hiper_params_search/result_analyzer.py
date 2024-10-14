import math
import os
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from tabulate import tabulate


class SearchResultAnalyzer:

    def __init__(self, output_dir: str, result_count: int = 10, save: bool = True):
        self.result_count = result_count
        self.output_dir = output_dir
        self.save = save

        self.__create_output_dir()

    def show_boxplot_graphics_from_tested_params(self, cv_results):
        df = pd.DataFrame(cv_results)

        param_columns = [col for col in df.columns if col.startswith('param_')]

        num_params = len(param_columns)
        num_windows = math.ceil(num_params / 4)

        directory = self.get_date_directory()

        for window_index in range(num_windows):
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))

            for param_index in range(4):
                index = window_index * 4 + param_index

                if index < num_params:
                    x = param_columns[index]
                    y = 'mean_test_score'

                    sns.scatterplot(data=df, x=x, y=y, ax=axes[param_index // 2, param_index % 2])

            if directory is not None:
                file_name = os.path.join(directory, f'scatter_plot_{window_index}.png')
                plt.savefig(file_name)

        plt.show()

    def get_date_directory(self):
        if self.save:
            directory = os.path.join(self.output_dir, self.__format_current_datetime())
            os.makedirs(directory)

            return directory
        else:
            return None

    def get_dataframe_bad_params(self, cv_results) -> DataFrame:
        df = pd.DataFrame(cv_results)
        df = df.sort_values(by='mean_test_score', ascending=False)

        return df[['mean_test_score', 'std_test_score', 'params']].tail(self.result_count)

    def get_dataframe_best_params(self, cv_results) -> DataFrame:
        df = pd.DataFrame(cv_results)
        df = df.sort_values(by='mean_test_score', ascending=False)

        return df[['mean_test_score', 'std_test_score', 'params']].head(self.result_count)

    def show_bad_params(self, cv_results):
        df = self.get_dataframe_bad_params(cv_results)
        print(tabulate(df, headers='keys', tablefmt='psql', colalign='left'))

    def show_best_params(self, cv_results):
        df = self.get_dataframe_best_params(cv_results)
        print(tabulate(df, headers='keys', tablefmt='psql', colalign='left'))

    def __create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def __format_current_datetime():
        now = datetime.now()
        return now.strftime("%d_%m_%y_%H_%M_%S")
