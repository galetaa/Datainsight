from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats as stats


class DataViewer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_overall(self) -> Dict[str, Any]:
        info = {
            # Существующие поля
            "shape": list(self.df.shape),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_count": {col: int(count) for col, count in self.df.isna().sum().items()},
            "missing_percent": {col: float(percent) for col, percent in (self.df.isna().mean() * 100).items()},
            "duplicates": int(self.df.duplicated().sum()),

            # Новые поля
            "memory_usage": {
                "total": self.df.memory_usage(deep=True).sum() / 1024 / 1024,  # в МБ
                "per_column": {col: self.df[col].memory_usage(deep=True) / 1024 / 1024 for col in self.df.columns}
            },
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns),
            "unique_values_count": {col: self.df[col].nunique() for col in self.df.columns},
            "value_ranges": {
                col: {
                    "min": self.df[col].min() if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                    "max": self.df[col].max() if pd.api.types.is_numeric_dtype(self.df[col]) else None
                } for col in self.df.columns
            }
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

    def sort_dataframe(self, column: str, ascending: bool = True) -> pd.DataFrame:

        if column not in self.df.columns:
            raise ValueError(f"Столбец {column} не найден")

        sorted_df = self.df.sort_values(by=column, ascending=ascending)

        return sorted_df

    def apply_sorting(self, column: str, ascending: bool = True):

        self.df = self.sort_dataframe(column, ascending)

    def statistical_tests(self, columns: List[str] = None) -> Dict[str, Any]:
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns

        test_results = {}

        for col in columns:
            # Инициализируем словарь для каждого столбца
            test_results[col] = {}

            # Очищаем данные от пропусков
            clean_data = self.df[col].dropna()

            if len(clean_data) == 0:
                test_results[col] = {
                    'error': 'Недостаточно данных для анализа'
                }
                continue

            # Тест Шапиро-Уилка на нормальность распределения
            try:
                _, shapiro_p = stats.shapiro(clean_data)
            except Exception as e:
                shapiro_p = None

            # Статистические показатели
            try:
                skewness = stats.skew(clean_data)
                kurtosis = stats.kurtosis(clean_data)

                # Квартили
                q1, q2, q3 = clean_data.quantile([0.25, 0.5, 0.75])
                iqr = q3 - q1

                test_results[col] = {
                    'shapiro_test': {
                        'statistic': shapiro_p,
                        'is_normal_distribution': shapiro_p > 0.05 if shapiro_p is not None else None
                    },
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'quartiles': {
                        'Q1': q1,
                        'median': q2,
                        'Q3': q3,
                        'IQR': iqr
                    }
                }
            except Exception as e:
                test_results[col] = {
                    'error': str(e)
                }

        return test_results

    def correlation_analysis(self, method: str = 'pearson') -> Dict[str, Any]:
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Корреляционные матрицы
        corr_matrices = {
            'pearson': self.df[numeric_columns].corr(method='pearson'),
            'spearman': self.df[numeric_columns].corr(method='spearman'),
            'kendall': self.df[numeric_columns].corr(method='kendall')
        }

        # Поиск сильных корреляций
        strong_correlations = []
        for matrix_type, corr_matrix in corr_matrices.items():
            matrix_correlations = []
            for i in range(len(numeric_columns)):
                for j in range(i + 1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    corr_value = corr_matrix.loc[col1, col2]

                    if abs(corr_value) > 0.7:
                        matrix_correlations.append({
                            'columns': (col1, col2),
                            'correlation': corr_value,
                            'correlation_type': 'positive' if corr_value > 0 else 'negative',
                            'matrix_type': matrix_type.capitalize()
                        })

            strong_correlations.extend(matrix_correlations)

        return {
            'correlation_matrices': {k: v.to_dict() for k, v in corr_matrices.items()},
            'strong_correlations': strong_correlations
        }

    def hypothesis_testing(self, column1: str, column2: str = None) -> Dict[str, Any]:
        results = {}

        # Существующие тесты
        if column2 is None:
            t_statistic, p_value = stats.ttest_1samp(self.df[column1].dropna(), popmean=0)
            results['one_sample_ttest'] = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'null_hypothesis_rejected': p_value < 0.05
            }
        else:
            t_statistic, p_value = stats.ttest_ind(
                self.df[column1].dropna(),
                self.df[column2].dropna()
            )
            results['two_sample_ttest'] = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'null_hypothesis_rejected': p_value < 0.05,
                'mean_difference_significant': p_value < 0.05
            }

        # Дополнительные статистические тесты
        if column2:
            # Тест Манна-Уитни (непараметрический)
            u_statistic, u_p_value = stats.mannwhitneyu(
                self.df[column1].dropna(),
                self.df[column2].dropna()
            )
            results['mann_whitney_test'] = {
                'u_statistic': u_statistic,
                'p_value': u_p_value,
                'statistically_significant': u_p_value < 0.05
            }

            # Тест Левене для проверки однородности дисперсий
            levene_statistic, levene_p_value = stats.levene(
                self.df[column1].dropna(),
                self.df[column2].dropna()
            )
            results['levene_test'] = {
                'levene_statistic': levene_statistic,
                'p_value': levene_p_value,
                'variances_equal': levene_p_value > 0.05
            }

        return results


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
