from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from typing import Literal


class Cleaner:
    def __init__(self, data: DataFrame | None = None):
        self.data = data

    def load_data(self, data: DataFrame):
        self.data = data

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

    def drop_duplicates(self, column: str | None = None, keep: Literal['first', 'last', False] = 'first'):
        df = self.data
        if column:
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден в DataFrame.")
            return df.drop_duplicates(subset=[column], keep=keep)
        else:
            return df.drop_duplicates(keep=keep)
