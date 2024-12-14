import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch
from sklearn.metrics import mean_squared_error, r2_score

from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from predictor.base_model import BaseMLModel


class LinearRegressionModel(BaseMLModel):
    def __init__(self, target_column=None, fit_intercept=True, copy_X=True, n_jobs=None, positive=False, **kwargs):
        """
        Линейная регрессия с расширенными возможностями

        Параметры:
        - target_column: целевой столбец для обучения модели
        - fit_intercept: включать ли свободный член
        - copy_X: копировать ли входные данные
        """
        super().__init__(model_type='regression', **kwargs)
        self.target_column = target_column

        # Параметры модели по умолчанию
        self.model_params = {
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'n_jobs': n_jobs,
            'positive': positive
        }

        # Экземпляр модели sklearn
        self.model = LinearRegression(**self.model_params)

    def fit(self, X=None, y=None):
        """
        Обучение модели с расширенной функциональностью
        """
        # Использование данных из реестра, если не переданы явно
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        # Обучение модели
        self.model.fit(X, y)

        # Логирование результатов
        self.logger.info("Linear Regression model trained")

        # Сохранение коэффициентов и метрик
        self.model_info = {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }

        return self

    def predict(self, X=None):
        """
        Прогнозирование с поддержкой различных входных данных
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик
        X_test, y_test = self.data_registry['test_data']
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.metrics['test_scores'] = {
            'mse': mse,
            'r2': r2
        }

        self.logger.info(f"Prediction metrics - MSE: {mse}, R2: {r2}")

        return predictions


class PolynomialRegressionModel(BaseMLModel):
    def __init__(self, degree=2, **kwargs):
        """
        Полиномиальная регрессия

        Параметры:
        - degree: степень полинома
        """
        super().__init__(model_type='regression', **kwargs)

        self.degree = degree

        # Создание пайплайна с полиномиальными признаками и линейной регрессией
        self.model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=self.degree)),
            ('linear_regression', LinearRegression())
        ])

    def fit(self, X=None, y=None):
        """
        Обучение полиномиальной модели
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        self.model.fit(X, y)

        self.logger.info(f"Polynomial Regression (degree {self.degree}) trained")

        return self

    def predict(self, X=None):
        """
        Прогнозирование для полиномиальной регрессии
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик
        X_test, y_test = self.data_registry['test_data']
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.metrics['test_scores'] = {
            'mse': mse,
            'r2': r2
        }

        self.logger.info(f"Prediction metrics - MSE: {mse}, R2: {r2}")

        return predictions


class RidgeRegressionModel(BaseMLModel):
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4, solver="auto",
                 positive=False, random_state=None, **kwargs):
        """
        Гребневая регрессия (Ridge Regression)

        Параметры:
        - alpha: параметр регуляризации
        """
        super().__init__(model_type='regression', **kwargs)

        self.model_params = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'positive': positive,
            'random_state': random_state,
        }

        self.model = Ridge(**self.model_params)

    def fit(self, X=None, y=None):
        """
        Обучение Ridge модели
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        self.model.fit(X, y)

        self.logger.info(f"Ridge Regression (alpha={self.model_params['alpha']}) trained")

        return self

    def predict(self, X=None):
        """
        Прогнозирование для Ridge Regression
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик
        X_test, y_test = self.data_registry['test_data']
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.metrics['test_scores'] = {
            'mse': mse,
            'r2': r2
        }

        self.logger.info(f"Prediction metrics - MSE: {mse}, R2: {r2}")

        return predictions


class DecisionTreeModel(BaseMLModel):
    def __init__(
            self,
            task_type='regression',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            **kwargs
    ):
        """
        Универсальная модель дерева решений для регрессии и классификации

        Параметры:
        - task_type: тип задачи (regression/classification)
        - max_depth: максимальная глубина дерева
        - min_samples_split: минимальное число сэмплов для сплита
        - min_samples_leaf: минимальное число сэмплов в листе
        """
        super().__init__(model_type=task_type, **kwargs)

        self.task_type = task_type
        self.model_params = {
            'criterion': "squared_error" if task_type == 'regression' else "gini",
            'splitter': "best",
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'random_state': None,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'ccp_alpha': 0.0,
            'monotonic_cst': None
        }

        # Выбор модели в зависимости от типа задачи
        self.model = (
            DecisionTreeRegressor(**self.model_params)
            if task_type == 'regression'
            else DecisionTreeClassifier(**self.model_params)
        )

    def fit(self, X=None, y=None):
        """
        Обучение модели дерева решений
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        self.model.fit(X, y)

        self.logger.info(f"Decision Tree ({self.task_type}) trained")

        # Сохранение важности признаков
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_

        return self

    def predict(self, X=None):
        """
        Прогнозирование с вычислением метрик
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик в зависимости от типа задачи
        X_test, y_test = self.data_registry['test_data']

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.metrics['test_scores'] = {
                'mse': mse,
                'r2': r2
            }

            self.logger.info(f"Regression Metrics - MSE: {mse}, R2: {r2}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)

            self.metrics['test_scores'] = {
                'accuracy': accuracy,
                'report': report
            }

            self.logger.info(f"Classification Metrics - Accuracy: {accuracy}")

        return predictions

    def visualize_tree(self, save_path='decision_tree.png'):
        """
        Визуализация дерева решений
        """
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            filled=True,
            feature_names=self.data_registry['raw_data'].columns.tolist()[:-1],  # Исключаем целевой столбец
            class_names=True if self.task_type == 'classification' else False
        )
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"Decision tree visualization saved to {save_path}")

    def feature_importance_plot(self, save_path='feature_importance.png'):
        """
        Построение графика важности признаков
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances are not available. Please fit the model first.")

        feature_names = self.data_registry['raw_data'].columns.tolist()[:-1]  # Исключаем целевой столбец

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=self.feature_importances,
            y=feature_names
        )
        plt.title('Важность признаков в дереве решений')
        plt.xlabel('Важность')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"Feature importance plot saved to {save_path}")


class RandomForestModel(BaseMLModel):
    def __init__(
            self,
            task_type='regression',
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            **kwargs
    ):
        """
        Универсальная модель случайного леса

        Параметры:
        - task_type: тип задачи (regression/classification)
        - n_estimators: количество деревьев в ансамбле
        - max_depth: максимальная глубина деревьев
        - min_samples_split: минимальное число сэмплов для сплита
        - min_samples_leaf: минимальное число сэмплов в листе
        """
        super().__init__(model_type=task_type, **kwargs)

        self.task_type = task_type
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': 42
        }

        # Выбор модели в зависимости от типа задачи
        self.model = (
            RandomForestRegressor(**self.model_params)
            if task_type == 'regression'
            else RandomForestClassifier(**self.model_params)
        )

    def fit(self, X=None, y=None):
        """
        Обучение модели случайного леса
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        self.model.fit(X, y)

        self.logger.info(f"Random Forest ({self.task_type}) trained")

        # Сохранение важности признаков
        self.feature_importances = self.model.feature_importances_

        return self

    def predict(self, X=None):
        """
        Прогнозирование с вычислением метрик
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик в зависимости от типа задачи
        X_test, y_test = self.data_registry['test_data']

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.metrics['test_scores'] = {
                'mse': mse,
                'r2': r2
            }

            self.logger.info(f"Regression Metrics - MSE: {mse}, R2: {r2}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.metrics['test_scores'] = {
                'accuracy': accuracy,
                'report': report
            }

            self.logger.info(f"Classification Metrics - Accuracy: {accuracy}")

        return predictions

    def grid_search_optimization(
            self,
            param_grid=None,
            cv=5,
            scoring='r2'
    ):
        """
        Оптимизация гиперпараметров через GridSearchCV

        Параметры:
        - param_grid: словарь параметров для поиска
        - cv: количество fold для кросс-валидации
        - scoring: метрика оценки качества
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring
        )

        X_train, y_train = self.data_registry['train_data']
        grid_search.fit(X_train, y_train)

        # Обновление модели лучшими параметрами
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        self.logger.info(f"Best Parameters: {self.best_params}")
        self.logger.info(f"Best Score: {grid_search.best_score_}")

        return grid_search

    def feature_importance_plot(self, save_path='rf_feature_importance.png'):
        """
        Построение графика важности признаков
        """
        feature_names = self.data_registry['raw_data'].columns.tolist()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=self.feature_importances,
            y=feature_names
        )
        plt.title('Важность признаков в случайном лесу')
        plt.xlabel('Важность')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"Random Forest feature importance plot saved to {save_path}")


class GradientBoostingModel(BaseMLModel):
    def __init__(
            self,
            task_type='regression',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            **kwargs
    ):
        """
        Универсальная модель градиентного бустинга

        Параметры:
        - task_type: тип задачи (regression/classification)
        - n_estimators: количество деревьев
        - learning_rate: скорость обучения
        - max_depth: максимальная глубина деревьев
        """
        super().__init__(model_type=task_type, **kwargs)

        self.task_type = task_type
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }

        # Выбор модели в зависимости от типа задачи
        self.model = (
            GradientBoostingRegressor(**self.model_params)
            if task_type == 'regression'
            else GradientBoostingClassifier(**self.model_params)
        )

    def fit(self, X=None, y=None):
        """
        Обучение модели градиентного бустинга
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        self.model.fit(X, y)

        self.logger.info(f"Gradient Boosting ({self.task_type}) trained")

        # Сохранение важности признаков
        self.feature_importances = self.model.feature_importances_

        return self

    def predict(self, X=None):
        """
        Прогнозирование с вычислением метрик
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        predictions = self.model.predict(X)

        # Вычисление метрик в зависимости от типа задачи
        X_test, y_test = self.data_registry['test_data']

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.metrics['test_scores'] = {
                'mse': mse,
                'r2': r2
            }

            self.logger.info(f"Regression Metrics - MSE: {mse}, R2: {r2}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.metrics['test_scores'] = {
                'accuracy': accuracy,
                'report': report
            }

            self.logger.info(f"Classification Metrics - Accuracy: {accuracy}")

        return predictions


class SVMModel(BaseMLModel):
    def __init__(
            self,
            task_type='regression',
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            **kwargs
    ):
        """
        Универсальная модель Support Vector Machine

        Параметры:
        - task_type: тип задачи (regression/classification)
        - kernel: тип ядра (linear, poly, rbf)
        - C: параметр регуляризации
        - epsilon: margin для SVR
        """
        super().__init__(model_type=task_type, **kwargs)

        self.task_type = task_type
        self.model_params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'random_state': 42
        }

        # Выбор модели в зависимости от типа задачи
        self.model = (
            SVR(**self.model_params)
            if task_type == 'regression'
            else SVC(**self.model_params)
        )

        # Масштабирование данных для SVM
        self.scaler = StandardScaler()

    def fit(self, X=None, y=None):
        """
        Обучение SVM модели
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        # Масштабирование данных
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

        self.logger.info(f"SVM ({self.task_type}) trained")

        return self

    def predict(self, X=None):
        """
        Прогнозирование с вычислением метрик
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        # Масштабирование тестовых данных
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        # Вычисление метрик в зависимости от типа задачи
        X_test, y_test = self.data_registry['test_data']

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.metrics['test_scores'] = {
                'mse': mse,
                'r2': r2
            }

            self.logger.info(f"Regression Metrics - MSE: {mse}, R2: {r2}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.metrics['test_scores'] = {
                'accuracy': accuracy,
                'report': report
            }

            self.logger.info(f"Classification Metrics - Accuracy: {accuracy}")

        return predictions


class KNNModel(BaseMLModel):
    def __init__(
            self,
            task_type='regression',
            n_neighbors=5,
            weights='uniform',
            **kwargs
    ):
        """
        Универсальная модель K-Nearest Neighbors

        Параметры:
        - task_type: тип задачи (regression/classification)
        - n_neighbors: количество соседей
        - weights: вес соседей (uniform/distance)
        """
        super().__init__(model_type=task_type, **kwargs)

        self.task_type = task_type
        self.model_params = {
            'n_neighbors': n_neighbors,
            'weights': weights
        }

        # Выбор модели в зависимости от типа задачи
        self.model = (
            KNeighborsRegressor(**self.model_params)
            if task_type == 'regression'
            else KNeighborsClassifier(**self.model_params)
        )

        # Масштабирование данных для KNN
        self.scaler = StandardScaler()

    def fit(self, X=None, y=None):
        """
        Обучение KNN модели
        """
        if X is None or y is None:
            X, y = self.data_registry['train_data']

        # Масштабирование данных
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

        self.logger.info(f"KNN ({self.task_type}) trained")

        return self

    def predict(self, X=None):
        """
        Прогнозирование с вычислением метрик
        """
        if X is None:
            X = self.data_registry['test_data'][0]

        # Масштабирование тестовых данных
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        # Вычисление метрик в зависимости от типа задачи
        X_test, y_test = self.data_registry['test_data']

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.metrics['test_scores'] = {
                'mse': mse,
                'r2': r2
            }

            self.logger.info(f"Regression Metrics - MSE: {mse}, R2: {r2}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.metrics['test_scores'] = {
                'accuracy': accuracy,
                'report': report
            }

            self.logger.info(f"Classification Metrics - Accuracy: {accuracy}")

        return predictions


class TimeSeriesModel(BaseMLModel):
    def __init__(
            self,
            forecast_horizon=1,
            model_type='exponential_smoothing',
            **kwargs
    ):
        """
        Модель прогнозирования временных рядов

        Параметры:
        - forecast_horizon: горизонт прогноза
        - model_type: тип модели (exponential_smoothing, arima, auto_arima)
        """
        super().__init__(model_type='time_series', **kwargs)

        self.time_series_data = None
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        self.model = None

    def prepare_time_series(self, data, date_column=None, target_column=None):
        """
        Подготовка данных временного ряда
        """
        if date_column:
            data = data.set_index(date_column)

        if target_column:
            data = data[target_column]

        self.time_series_data = data

        return data

    def fit(self):
        """
        Обучение модели временных рядов
        """
        if self.model_type == 'exponential_smoothing':
            self.model = ExponentialSmoothing(
                self.time_series_data,
                trend='add',
                seasonal='add',
                seasonal_periods=12
            ).fit()

        elif self.model_type == 'arima':
            self.model = ARIMA(
                self.time_series_data,
                order=(1, 1, 1)
            ).fit()

        elif self.model_type == 'auto_arima':
            self.model = pm.auto_arima(
                self.time_series_data,
                seasonal=True,
                m=12
            )

        self.logger.info(f"Time Series Model ({self.model_type}) trained")

        return self

    def predict(self):
        """
        Прогноз временного ряда
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict()")

        predictions = self.model.forecast(steps=self.forecast_horizon)

        # Визуализация прогноза
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_series_data, label='Исторические данные')
        plt.plot(
            pd.date_range(
                start=self.time_series_data.index[-1],
                periods=self.forecast_horizon + 1
            )[1:],
            predictions,
            label='Прогноз',
            color='red'
        )
        plt.title(f'Прогноз методом {self.model_type}')
        plt.legend()
        plt.savefig('time_series_forecast.png')
        plt.close()

        self.logger.info(f"Time Series Forecast: {predictions}")

        return predictions

    def model_diagnostics(self):
        """
        Диагностика модели временных рядов
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Остатки модели
        residuals = self.model.resid

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(residuals)
        plt.title('Остатки')

        plt.subplot(2, 2, 2)
        sns.histplot(residuals, kde=True)
        plt.title('Распределение остатков')

        plt.tight_layout()
        plt.savefig('time_series_diagnostics.png')
        plt.close()

        return residuals
