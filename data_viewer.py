import pandas as pd
import numpy as np
from typing import Dict, Any


class DataViewer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_overall(self) -> Dict[str, Any]:
        """Общая информация о датасете."""
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.apply(lambda x: str(x)).to_dict(),
            "missing_count": self.df.isna().sum().to_dict(),
            "missing_percent": (self.df.isna().mean() * 100).to_dict(),
            "duplicates": self.df.duplicated().sum()
        }
        return info

    def analyze_numeric(self, col: str) -> Dict[str, Any]:
        """Метрики для числового столбца."""
        series = self.df[col].dropna()
        if series.empty:
            return {}
        desc = series.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()

        # Дополнительные метрики
        skewness = series.skew()
        kurt = series.kurt()

        # Пример определения выбросов по IQR
        q1, q3 = desc['25%'], desc['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)].count()

        return {
            "count": desc["count"],
            "mean": desc["mean"],
            "std": desc["std"],
            "min": desc["min"],
            "25%": desc["25%"],
            "50% (median)": desc["50%"],
            "75%": desc["75%"],
            "max": desc["max"],
            "skewness": skewness,
            "kurtosis": kurt,
            "outliers_count": outliers
        }

    def analyze_categorical(self, col: str) -> Dict[str, Any]:
        """Метрики для категориального столбца."""
        series = self.df[col].dropna().astype(str)
        if series.empty:
            return {}

        value_counts = series.value_counts()
        top_values = value_counts.head(10).to_dict()

        # Подсчёт энтропии распределения категорий
        probs = value_counts / value_counts.sum()
        entropy = -(probs * np.log2(probs)).sum()

        return {
            "unique_count": value_counts.size,
            "top_values": top_values,
            "entropy": entropy,
            "most_common_category": value_counts.index[0],
            "most_common_count": value_counts.iloc[0]
        }

    def analyze_datetime(self, col: str) -> Dict[str, Any]:
        """Метрики для временного столбца."""
        if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
            # Попытка преобразовать к datetime, если возможно
            try:
                dt_series = pd.to_datetime(self.df[col], errors='coerce').dropna()
            except Exception as err:
                return {}
        else:
            dt_series = self.df[col].dropna()

        if dt_series.empty:
            return {}

        min_date = dt_series.min()
        max_date = dt_series.max()
        date_range = max_date - min_date

        # Подсчёт распределения по дням недели
        weekdays_counts = dt_series.dt.day_name().value_counts().to_dict()

        return {
            "min_date": str(min_date),
            "max_date": str(max_date),
            "date_range": str(date_range),
            "count": dt_series.size,
            "weekdays_distribution": weekdays_counts
        }

    def analyze_text(self, col: str) -> Dict[str, Any]:
        """Метрики для текстового столбца."""
        series = self.df[col].dropna().astype(str)
        if series.empty:
            return {}

        lengths = series.apply(len)

        # Статистика по длинам строк
        length_desc = lengths.describe().to_dict()

        # Частотный словарь
        # Простой подход: разобьём по пробелам (зависит от специфики данных)
        word_counts = {}
        for text in series:
            words = text.split()
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Топ-10 слов
        word_counts_sorted = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = dict(word_counts_sorted[:10])

        return {
            "count": length_desc["count"],
            "mean_length": length_desc["mean"],
            "max_length": length_desc["max"],
            "min_length": length_desc["min"],
            "top_words": top_words,
            "unique_words_count": len(word_counts)
        }

    def analyze_missing(self) -> Dict[str, Any]:
        """Общая информация о пропусках."""
        missing = self.df.isna().sum()
        total_rows = len(self.df)
        return {
            "missing_counts": missing.to_dict(),
            "missing_percent": (missing / total_rows * 100).to_dict()
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Запустить полный набор анализов по типам данных."""
        result = {
            "overall": self.analyze_overall(),
            "columns": {}
        }

        for col in self.df.columns:
            col_type = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                result["columns"][col] = {
                    "type": "numeric",
                    "analysis": self.analyze_numeric(col)
                }
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                result["columns"][col] = {
                    "type": "datetime",
                    "analysis": self.analyze_datetime(col)
                }
            else:
                # Попытка определить, текст или категориальные данные
                # Здесь условно: если уникальных значений не очень много, считаем категориальным
                unique_count = self.df[col].nunique(dropna=True)
                if unique_count < (0.3 * len(self.df)) and unique_count < 1000:
                    # Простая эвристика для категориального
                    result["columns"][col] = {
                        "type": "categorical",
                        "analysis": self.analyze_categorical(col)
                    }
                else:
                    # Иначе рассматриваем как текст
                    result["columns"][col] = {
                        "type": "text",
                        "analysis": self.analyze_text(col)
                    }

        return result


# Пример использования:
if __name__ == "__main__":
    # Примерный DataFrame
    data = {
        "id": [1, 2, 3, 4, 5],
        "age": [25, 30, 22, np.nan, 40],
        "category": ["A", "B", "A", "C", "A"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-10", None, "2021-02-01"]),
        "description": ["This is a sample", "Another text", "Text data here", "", "Sample again"]
    }
    df_example = pd.DataFrame(data)

    analyzer = DataViewer(df_example)
    full_report = analyzer.run_full_analysis()
    # Посмотреть отчет
    from pprint import pprint

    pprint(full_report)
