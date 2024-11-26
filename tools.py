from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas import DataFrame
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder


class BaseTool:
    def __init__(self, data: DataFrame | None = None):
        self.data = data

    def load_data(self, data: DataFrame):
        self.data = data


class Cleaner(BaseTool):
    def find_missing_values(self, column: str | None = None, only_nan: bool = False):
        df: DataFrame = self.data
        whitespace_mask: DataFrame = DataFrame()
        if not (column is None):
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")
            nan_mask = df[column].isna()
            if not only_nan:
                whitespace_mask = df[column].apply(lambda x: isinstance(x, str) and x.strip() == "")
            mask = nan_mask | whitespace_mask
            return mask
        else:
            nan_mask: DataFrame = df.isna()
            if not only_nan:
                whitespace_mask = df.applymap(lambda x: isinstance(x, str) and x.strip() == "")
            mask = whitespace_mask | nan_mask
            return mask

    def clean_missing_values(self, column: str | None = None, only_nan: bool = False):
        df: DataFrame = self.data
        if not (column is None):
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")

        mask = self.find_missing_values(column=column, only_nan=only_nan)

        df = df.loc[~mask.any(axis=1)]
        return df

    @staticmethod
    def __get_fill_value(series: Series,
                         method: Literal["mean", "median", "min", "max", "mode", "constant"],
                         value: int | float | complex | np.int32 | np.int64 | np.uint32 | np.uint64 |
                                np.float32 | np.float64 | np.complex64 | np.complex128 = None):
        if method == "mean":
            return series.mean()
        elif method == "median":
            return series.median()
        elif method == "min":
            return series.min()
        elif method == "max":
            return series.max()
        elif method == "mode":
            return series.mode().iloc[0] if not series.mode().empty else np.nan
        elif method == "constant":
            if value is None:
                raise ValueError("Для метода 'constant' требуется параметр `value`.")
            return value
        else:
            raise ValueError(
                f"Неверный метод заполнения: '{method}'. Доступные методы: mean, median, min, max, mode, constant.")

    def fill_numeric_missing(self, column: str | None = None,
                             method: Literal["mean", "median", "min", "max", "mode", "constant"] = "mean",
                             value: int | float | complex | np.int32 | np.int64 | np.uint32 | np.uint64 |
                                    np.float32 | np.float64 | np.complex64 | np.complex128 = None):
        df: DataFrame = self.data

        if column:
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")

            if pd.api.types.is_numeric_dtype(df[column]):
                fill_value = self.__get_fill_value(df[column], method, value)
                df[column] = df[column].fillna(fill_value)
            else:
                df[column] = df[column]
        else:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = self.__get_fill_value(df[col], method, value)
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col]
        return df

    def find_duplicates(self, column: str | None = None, keep: Literal['first', 'last', False] = 'first'):
        df = self.data
        if column:
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")
            duplicate_mask = df.duplicated(subset=[column], keep=keep)
        else:
            duplicate_mask = df.duplicated(keep=keep)
        return df.loc[duplicate_mask]

    def clean_duplicates(self, column: str | None = None, keep: Literal['first', 'last', False] = 'first'):
        df = self.data
        if column:
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")
            return df.drop_duplicates(subset=[column], keep=keep)
        else:
            return df.drop_duplicates(keep=keep)


class AnomaliesDetector(BaseTool):
    def detect_anomalies_3sigma(self, column):
        df = self.data

        mean = df[column].mean()
        std = df[column].std()
        anomalies = df[(df[column] < mean - 3 * std) | (df[column] > mean + 3 * std)]
        return anomalies

    def detect_anomalies_iqr(self, column):
        df = self.data

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        anomalies = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
        return anomalies

    def detect_anomalies_zscore(self, column, threshold=3):
        df = self.data

        df['z_score'] = zscore(df[column])
        anomalies = df[df['z_score'].abs() > threshold]
        return anomalies.drop(columns=['z_score'])

    def detect_anomalies_isolation_forest(self, columns):
        df = self.data

        model = IsolationForest(random_state=42)
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_svm(self, columns):
        df = self.data

        model = OneClassSVM(kernel='rbf', gamma='auto')
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_dbscan(self, columns, eps=0.5, min_samples=5):
        df = self.data

        model = DBSCAN(eps=eps, min_samples=min_samples)
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_knn(self, columns, n_neighbors=5, threshold=1.5):
        df = self.data

        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(df[columns])
        distances, _ = knn.kneighbors(df[columns])
        avg_distances = distances.mean(axis=1)
        df['distance'] = avg_distances
        anomalies = df[df['distance'] > threshold]
        return anomalies.drop(columns=['distance'])

    def detect_anomalies_stl(self, column, period, threshold=1.5):
        df = self.data

        decomposition = seasonal_decompose(df[column], period=period)
        residual = decomposition.resid
        df['residual'] = residual
        anomalies = df[residual.abs() > threshold]
        return anomalies.drop(columns=['residual'])

    def detect_anomalies(self, method, column=None, columns=None, **kwargs):
        if method == '3sigma':
            return self.detect_anomalies_3sigma(column)
        elif method == 'iqr':
            return self.detect_anomalies_iqr(column)
        elif method == 'zscore':
            return self.detect_anomalies_zscore(column, **kwargs)
        elif method == 'isolation_forest':
            return self.detect_anomalies_isolation_forest(columns)
        elif method == 'svm':
            return self.detect_anomalies_svm(columns)
        elif method == 'dbscan':
            return self.detect_anomalies_dbscan(columns, **kwargs)
        elif method == 'knn':
            return self.detect_anomalies_knn(columns, **kwargs)
        elif method == 'stl':
            return self.detect_anomalies_stl(column, **kwargs)
        else:
            raise ValueError(f"Метод '{method}' не поддерживается.")


class Normalizer(BaseTool):
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


class Converter(BaseTool):
    def convert_to_numeric(self, columns=None, errors='ignore'):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=['object']).columns  # Столбцы с объектами (строками)

        for col in columns:
            # Попробуем преобразовать столбец в числовой тип
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')  # Преобразуем в числовой формат
            except ValueError:
                if errors == 'raise':
                    raise ValueError(f"Невозможно преобразовать столбец '{col}' в числовой тип.")
                else:
                    # Если ошибка, пропускаем этот столбец
                    pass

        return df

    def convert_to_datetime(self, columns=None, formate=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=['object']).columns  # Столбцы с датами в виде строк

        for col in columns:
            df[col] = pd.to_datetime(df[col], format=formate, errors='coerce')  # Преобразуем в datetime
        return df

    def one_hot_encode(self, columns=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=['object']).columns

        df = pd.get_dummies(df, columns=columns, drop_first=True)  # One-hot кодировка
        return df

    def label_encode(self, columns=None):
        df = self.data

        if columns is None:
            columns = df.select_dtypes(include=['object']).columns

        label_encoder = LabelEncoder()
        for col in columns:
            df[col] = label_encoder.fit_transform(df[col])  # Преобразуем в метки
        return df

    def optimize_data_types(self):
        df = self.data

        for col in df.columns:
            dtype = df[col].dtype
            if dtype == 'float64':
                df[col] = df[col].astype('float32')  # Преобразуем в меньший тип данных
            elif dtype == 'int64':
                if df[col].max() < 2 ** 31 - 1:
                    df[col] = df[col].astype('int32')  # Преобразуем в меньший тип
                elif df[col].max() < 2 ** 15 - 1:
                    df[col] = df[col].astype('int16')
            elif dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:  # Если уникальных значений меньше 50%
                    df[col] = df[col].astype('category')  # Преобразуем строковые столбцы в категориальные
        return df
