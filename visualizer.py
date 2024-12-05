import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Union, Optional


class DataValidator:
    @staticmethod
    def validate_1d_data(data: Union[List, np.ndarray, pd.Series],
                         min_points: int = 2,
                         max_outlier_percentage: float = 10) -> dict:
        """
        Валидация одномерных данных

        :param data: Входные данные
        :param min_points: Минимальное количество точек
        :param max_outlier_percentage: Максимальный процент выбросов
        :return: Словарь с результатами валидации
        """
        # Преобразование к numpy массиву
        data_array = np.array(data, dtype=float)

        # Проверка типа данных
        validation_result = {
            'is_valid': True,
            'errors': []
        }

        # Проверка числовых данных
        if not np.issubdtype(data_array.dtype, np.number):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Данные должны быть числовыми")

        # Проверка количества точек
        if len(data_array) < min_points:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Недостаточно точек. Требуется минимум {min_points}")

        # Проверка NaN и бесконечных значений
        if np.isnan(data_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют NaN значения")

        if np.isinf(data_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют бесконечные значения")

        # Проверка выбросов (межквартильный размах)
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1

        # Критерий выбросов: значения вне 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
        outlier_percentage = len(outliers) / len(data_array) * 100

        if outlier_percentage > max_outlier_percentage:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Слишком много выбросов: {outlier_percentage:.2f}%")

        return validation_result

    @staticmethod
    def validate_2d_data(x: Union[List, np.ndarray, pd.Series],
                         y: Union[List, np.ndarray, pd.Series],
                         plot_type: str = 'scatter') -> dict:
        """
        Валидация двумерных данных

        :param x: Данные по X
        :param y: Данные по Y
        :param plot_type: Тип графика
        :return: Словарь с результатами валидации
        """
        # Преобразование к numpy массивам
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)

        validation_result = {
            'is_valid': True,
            'errors': []
        }

        # Проверка длины данных
        if len(x_array) != len(y_array):
            validation_result['is_valid'] = False
            validation_result['errors'].append("X и Y должны иметь одинаковую длину")

        # Проверка числовых данных
        if not (np.issubdtype(x_array.dtype, np.number) and
                np.issubdtype(y_array.dtype, np.number)):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Данные должны быть числовыми")

        # Проверка NaN и бесконечных значений
        if np.isnan(x_array).any() or np.isnan(y_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют NaN значения")

        if np.isinf(x_array).any() or np.isinf(y_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют бесконечные значения")

        # Специфические проверки для разных типов графиков
        if plot_type in ['bar', 'line']:
            # Для bar и line желательны дискретные или упорядоченные значения по X
            if not np.all(np.diff(x_array) > 0):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Для {plot_type} рекомендуется упорядоченность X")

        return validation_result

    @staticmethod
    def validate_3d_data(x: Optional[Union[List, np.ndarray, pd.Series]] = None,
                         y: Optional[Union[List, np.ndarray, pd.Series]] = None,
                         z: Union[List, np.ndarray, pd.Series] = None,
                         plot_type: str = 'scatter3d') -> dict:
        """
        Валидация трехмерных данных

        :param x: Данные по X
        :param y: Данные по Y
        :param z: Данные по Z
        :param plot_type: Тип графика
        :return: Словарь с результатами валидации
        """
        validation_result = {
            'is_valid': True,
            'errors': []
        }

        # Специфика для разных 3D графиков
        if plot_type in ['scatter3d']:
            # Преобразование к numpy массивам
            x_array = np.array(x, dtype=float)
            y_array = np.array(y, dtype=float)
            z_array = np.array(z, dtype=float)

            # Проверка длины данных
            if len(set(map(len, [x_array, y_array, z_array]))) > 1:
                validation_result['is_valid'] = False
                validation_result['errors'].append("X, Y и Z должны иметь одинаковую длину")

            # Проверка числовых данных
            if not (np.issubdtype(x_array.dtype, np.number) and
                    np.issubdtype(y_array.dtype, np.number) and
                    np.issubdtype(z_array.dtype, np.number)):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Данные должны быть числовыми")

            # Проверка NaN и бесконечных значений
            if (np.isnan(x_array).any() or np.isnan(y_array).any() or np.isnan(z_array).any()):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют NaN значения")

            if (np.isinf(x_array).any() or np.isinf(y_array).any() or np.isinf(z_array).any()):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют бесконечные значения")

        elif plot_type in ['surface', 'contour', 'heatmap']:
            # Преобразование к numpy массиву
            z_array = np.array(z)

            # Проверка 2D массива
            if z_array.ndim != 2:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Для surface/contour/heatmap требуется 2D массив")

            # Проверка числовых данных
            if not np.issubdtype(z_array.dtype, np.number):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Данные должны быть числовыми")

            # Проверка NaN и бесконечных значений
            if np.isnan(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют NaN значения")

            if np.isinf(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют бесконечные значения")

        return validation_result


class Visualizer:
    def __init__(self, visualization_type: str):
        """
        Инициализация визуализатора с определенным типом графика

        :param visualization_type: Тип визуализации
        """
        self.visualization_type = visualization_type
        self.x: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.y: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.z: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.validation_result: Optional[dict] = None

    def set_data(self, x=None, y=None, z=None):
        """
        Установка данных для визуализации

        :param x: Данные для оси X
        :param y: Данные для оси Y
        :param z: Данные для оси Z
        :return: self для цепочки вызовов
        """
        self.x = x
        self.y = y
        self.z = z
        return self

    def validate(self) -> bool:
        """
        Валидация данных перед построением графика

        :return: Результат валидации
        """
        try:
            if self.visualization_type in ['histogram', 'kde', 'distribution']:
                self.validation_result = DataValidator.validate_1d_data(self.x)

            elif self.visualization_type in ['scatter', 'line', 'bar', 'area', 'box', 'violin']:
                self.validation_result = DataValidator.validate_2d_data(
                    self.x, self.y, plot_type=self.visualization_type
                )

            elif self.visualization_type in ['scatter3d', 'surface', 'contour', 'heatmap']:
                self.validation_result = DataValidator.validate_3d_data(
                    self.x, self.y, self.z, plot_type=self.visualization_type
                )

            else:
                raise ValueError(f"Неподдерживаемый тип визуализации: {self.visualization_type}")

            return self.validation_result['is_valid']

        except Exception as e:
            self.validation_result = {
                'is_valid': False,
                'errors': [str(e)]
            }
            return False

    def plot(self, title: str = 'Visualization'):
        """
        Создание визуализации в зависимости от типа

        :param title: Заголовок графика
        :return: Plotly Figure
        :raises ValueError: Если данные не прошли валидацию
        """
        # Валидация данных перед построением
        if not self.validate():
            error_message = "\n".join(self.validation_result['errors'])
            raise ValueError(f"Ошибка валидации данных:\n{error_message}")

        # Далее весь код из предыдущей версии Visualizer
        # Методы _one_dimensional_plot, _two_dimensional_plot, _three_dimensional_plot
        # остаются без изменений

        if self.visualization_type in ['histogram', 'kde', 'distribution']:
            return self._one_dimensional_plot(title)

        elif self.visualization_type in ['scatter', 'line', 'bar', 'area', 'box', 'violin']:
            return self._two_dimensional_plot(title)

        elif self.visualization_type in ['scatter3d', 'surface', 'contour', 'heatmap']:
            return self._three_dimensional_plot(title)

    def _one_dimensional_plot(self, title):
        """Одномерные визуализации"""
        if self.visualization_type == 'histogram':
            fig = px.histogram(x=self.x, title=title)
        elif self.visualization_type == 'kde':
            fig = px.density_contour(x=self.x, title=title)
        elif self.visualization_type == 'distribution':
            fig = px.ecdf(x=self.x, title=title)
        else:
            raise ValueError("Некорректный тип графика")
        return fig

    def _two_dimensional_plot(self, title):
        """Двумерные визуализации"""
        if self.visualization_type == 'scatter':
            fig = px.scatter(x=self.x, y=self.y, title=title)
        elif self.visualization_type == 'line':
            fig = px.line(x=self.x, y=self.y, title=title)
        elif self.visualization_type == 'bar':
            fig = px.bar(x=self.x, y=self.y, title=title)
        elif self.visualization_type == 'area':
            fig = px.area(x=self.x, y=self.y, title=title)
        elif self.visualization_type == 'box':
            fig = px.box(x=self.x, y=self.y, title=title)
        elif self.visualization_type == 'violin':
            fig = px.violin(x=self.x, y=self.y, title=title)
        else:
            raise ValueError("Некорректный тип графика")
        return fig

    def _three_dimensional_plot(self, title):
        """Трехмерные визуализации"""
        if self.visualization_type == 'scatter3d':
            # Преобразование многомерных массивов в плоские
            if isinstance(self.x, np.ndarray) and self.x.ndim > 1:
                x = self.x.ravel()
                y = self.y.ravel()
                z = self.z.ravel()
            else:
                x, y, z = self.x, self.y, self.z

            fig = px.scatter_3d(x=x, y=y, z=z, title=title)

        elif self.visualization_type == 'surface':
            # Создание Surface plot с использованием gridded данных
            fig = go.Figure(data=[go.Surface(z=self.z)])
            fig.update_layout(title=title)

        elif self.visualization_type == 'contour':
            # Создание контурного графика
            fig = go.Figure(data=go.Contour(z=self.z))
            fig.update_layout(title=title)

        elif self.visualization_type == 'heatmap':
            # Создание тепловой карты
            fig = px.imshow(self.z, title=title)
        else:
            raise ValueError("Некорректный тип графика")

        return fig

    def show(self):
        """Отображение графика с предварительной валидацией"""
        fig = self.plot()
        fig.show()

    def save(self, filename='visualization.html'):
        """Сохранение графика с предварительной валидацией"""
        fig = self.plot()
        fig.write_html(filename)
