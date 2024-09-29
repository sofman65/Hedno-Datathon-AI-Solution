import pandas as pd
import numpy as np
import pickle

class ImbDataProcessor:
    def __init__(self, imbData):
        self.imbData = imbData
        self.grouped = self.imbData.groupby(['BS_RATE', 'XRHSH', 'ACCT_CONTROL', 'SUPPLIER', 'VOLTAGE', 'cluster_labels'])
        self.group_averages = {}
        self.processed_data = None

    @staticmethod
    def calculate_average(series):
        count_non_zeros = [0] * len(series.iloc[0])
        sum_values = [0] * len(series.iloc[0])

        for time_series in series:
            for i, value in enumerate(time_series):
                if value != 0:
                    count_non_zeros[i] += 1
                    sum_values[i] += value

        return [sum_val / count if count > 0 else 0 for sum_val, count in zip(sum_values, count_non_zeros)]

    def process(self):
        for name, group in list(self.grouped):
            avg_time_series = self.calculate_average(group['time_series'])
            self.group_averages[name] = avg_time_series
            for idx in group.index:
                self.imbData.at[idx, 'time_series'] = [self.imbData.at[idx, 'time_series'][i] if self.imbData.at[idx, 'time_series'][i] != 0 else avg_time_series[i] for i in range(len(self.imbData.at[idx, 'time_series']))]

        self.processed_data = self.imbData.copy()

    def get_processed_data(self):
        return self.processed_data

    def impute_new_data(self, new_data):
        new_data = new_data.copy()
        key = tuple(new_data[col].iloc[0] for col in ['BS_RATE', 'XRHSH', 'ACCT_CONTROL', 'SUPPLIER', 'VOLTAGE', 'cluster_labels'])

        if key in self.group_averages:
            avg_time_series = self.group_averages[key]
            new_data['time_series'] = new_data['time_series'].apply(lambda ts: [ts[i] if ts[i] != 0 else avg_time_series[i] for i in range(len(ts))])
        else:
            print(f"Warning: Group {key} not found. No imputation performed.")

        return new_data

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
