from pandas import DataFrame
import numpy as np


class Normalizer:
    def __init__(self, data: DataFrame | None = None):
        self.data = data

    def load_data(self, data: DataFrame):
        self.data = data

    def min_max_normalize(self, columns=None, feature_range=(0, 1)):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        min_val, max_val = feature_range
        df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
        df[columns] = df[columns] * (max_val - min_val) + min_val

        return df

    def z_score_normalize(self, columns=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
        return df

    def max_abs_normalize(self, columns=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        df[columns] = df[columns] / df[columns].abs().max()
        return df

    def robust_normalize(self, columns=None):
        df = self.data
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        # Применяем к каждому столбцу медиану и межквартильный размах
        for col in columns:
            median = df[col].median()
            IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
            df[col] = (df[col] - median) / IQR

        return df

    def log_transform(self, columns=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        # Применяем логарифм с добавлением небольшого сдвига для избежания ошибок с 0
        df[columns] = np.log1p(df[columns])  # log(1 + X), чтобы избежать логарифма от 0
        return df

    def normalize_data(self, method='zscore', columns=None, **kwargs):
        if method == 'min_max':
            return self.min_max_normalize(columns, **kwargs)
        elif method == 'zscore':
            return self.z_score_normalize(columns)
        elif method == 'max_abs':
            return self.max_abs_normalize(columns)
        elif method == 'robust':
            return self.robust_normalize(columns)
        elif method == 'log':
            return self.log_transform(columns)
        else:
            raise ValueError(f"Метод '{method}' не поддерживается.")
