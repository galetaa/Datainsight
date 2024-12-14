import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Any
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class BaseMLModel:
    def __init__(
            self,
            model_type: str = 'regression',
            data_preprocessing: Dict[str, Any] = None,
            gpu_acceleration: bool = True,
            logging_config: Dict[str, Any] = None
    ):
        """
        Инициализация базового класса машинного обучения

        Параметры:
        - model_type: тип задачи (regression, classification, clustering)
        - data_preprocessing: настройки предобработки данных
        - gpu_acceleration: использование GPU
        - logging_config: настройки логирования
        """
        self.model_type = model_type
        self.device = self._setup_device(gpu_acceleration)

        # Настройки предобработки по умолчанию
        self.preprocessing_config = data_preprocessing or {
            'scaling': 'standard',  # standard, minmax
            'imputation': 'mean',  # mean, median, mode
            'feature_selection': {
                'method': 'f_regression',
                'n_features': 10
            },
            'dimensionality_reduction': {
                'method': 'pca',
                'n_components': 0.95  # сохранять 95% дисперсии
            }
        }

        # Настройка логирования
        self._setup_logging(logging_config)

        # Хранение промежуточных данных и состояний
        self.data_registry = {
            'raw_data': None,
            'processed_data': None,
            'train_data': None,
            'test_data': None,
            'model_history': []
        }

        # Метрики и отчеты
        self.metrics = {
            'train_scores': [],
            'validation_scores': [],
            'test_scores': []
        }

    def _setup_device(self, gpu_acceleration: bool) -> torch.device:
        """
        Настройка устройства для вычислений (GPU/CPU)
        """
        if gpu_acceleration and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _setup_logging(self, logging_config: Dict[str, Any] = None):
        """
        Настройка системы логирования
        """
        log_config = logging_config or {
            'level': logging.INFO,
            'format': '%(asctime)s - %(levelname)s: %(message)s',
            'filename': 'ml_model.log'
        }

        logging.basicConfig(
            level=log_config['level'],
            format=log_config['format'],
            filename=log_config['filename']
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(
            self,
            data: Union[pd.DataFrame, np.ndarray, str],
            target_column: Optional[str] = None
    ):
        """
        Загрузка и первичный анализ данных

        Поддерживает:
        - pandas DataFrame
        - numpy array
        - путь к CSV/Excel файлу
        """
        if isinstance(data, str):
            if data.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(data)

        self.data_registry['raw_data'] = data
        self.target_column = target_column

        # Первичный анализ данных
        self._perform_eda()

    def _perform_eda(self):
        """
        Первичный разведочный анализ данных
        """
        raw_data = self.data_registry['raw_data']

        # Базовая статистика
        self.data_stats = {
            'total_rows': len(raw_data),
            'total_columns': len(raw_data.columns),
            'missing_values': raw_data.isnull().sum(),
            'data_types': raw_data.dtypes,
            'numeric_columns': raw_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': raw_data.select_dtypes(include=['object']).columns.tolist()
        }

        self.logger.info(f"EDA Summary: {self.data_stats}")

    def preprocess_data(self, custom_pipeline: Pipeline = None):
        """
        Комплексная предобработка данных с гибкими настройками

        Поддерживает:
        - Заполнение пропусков
        - Масштабирование
        - Отбор признаков
        - Понижение размерности
        """
        X = self.data_registry['raw_data'].drop(columns=[self.target_column])
        y = self.data_registry['raw_data'][self.target_column] if self.target_column else None

        if custom_pipeline:
            processed_data = custom_pipeline.fit_transform(X, y)
        else:
            # Создание стандартного пайплайна предобработки
            pipeline_steps = []

            # Заполнение пропусков
            if self.preprocessing_config['imputation'] == 'mean':
                pipeline_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif self.preprocessing_config['imputation'] == 'median':
                pipeline_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif self.preprocessing_config['imputation'] == 'mode':
                pipeline_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

            # Масштабирование
            if self.preprocessing_config['scaling'] == 'standard':
                pipeline_steps.append(('scaler', StandardScaler()))
            elif self.preprocessing_config['scaling'] == 'minmax':
                pipeline_steps.append(('scaler', MinMaxScaler()))

            # Отбор признаков
            if self.preprocessing_config['feature_selection']['method'] == 'f_regression':
                pipeline_steps.append((
                    'feature_selector',
                    SelectKBest(
                        score_func=f_regression,
                        k=self.preprocessing_config['feature_selection']['n_features']
                    )
                ))

            # Понижение размерности
            if self.preprocessing_config['dimensionality_reduction']['method'] == 'pca':
                pipeline_steps.append((
                    'pca',
                    PCA(n_components=self.preprocessing_config['dimensionality_reduction']['n_components'])
                ))

            preprocessing_pipeline = Pipeline(pipeline_steps)
            processed_data = preprocessing_pipeline.fit_transform(X, y)

        self.data_registry['processed_data'] = processed_data
        self.logger.info("Data preprocessing completed")

    def split_data(
            self,
            test_size: float = 0.2,
            stratify: Optional[str] = None,
            random_state: int = 42
    ):
        """
        Разделение данных на train/test с гибкими настройками

        Поддерживает стратификацию, фиксацию случайности
        """
        X = self.data_registry['processed_data']
        y = self.data_registry['raw_data'][self.target_column] if self.target_column else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )

        self.data_registry.update({
            'train_data': (X_train, y_train),
            'test_data': (X_test, y_test)
        })

        self.logger.info(f"Data split: Train {len(X_train)}, Test {len(X_test)}")

    def cross_validation(
            self,
            model,
            cv: int = 5,
            scoring: str = 'r2'
    ):
        """
        Кросс-валидация с расширенными возможностями
        """
        X_train, y_train = self.data_registry['train_data']

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=KFold(n_splits=cv, shuffle=True),
            scoring=scoring
        )

        self.metrics['cross_validation_scores'] = cv_scores
        self.logger.info(f"Cross-validation scores: {cv_scores}")

        return cv_scores