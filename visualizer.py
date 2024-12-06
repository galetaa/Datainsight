from typing import List, Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DataValidator:
    @staticmethod
    def validate_1d_data(data: Union[List, np.ndarray, pd.Series],
                         min_points: int = 2,
                         max_outlier_percentage: float = 10) -> dict:

        data_array = np.array(data, dtype=float)

        validation_result = {
            'is_valid': True,
            'errors': []
        }

        if not np.issubdtype(data_array.dtype, np.number):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Данные должны быть числовыми")

        if len(data_array) < min_points:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"Недостаточно точек. Требуется минимум {min_points}"
            )

        if np.isnan(data_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют NaN значения")

        if np.isinf(data_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют бесконечные значения")

        if len(data_array) > 0:
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
            outlier_percentage = len(outliers) / len(data_array) * 100 if len(data_array) > 0 else 0

            if outlier_percentage > max_outlier_percentage:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Слишком много выбросов: {outlier_percentage:.2f}%")

        return validation_result

    @staticmethod
    def validate_2d_data(x: Union[List, np.ndarray, pd.Series],
                         y: Union[List, np.ndarray, pd.Series],
                         plot_type: str = 'scatter') -> dict:

        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
        validation_result = {
            'is_valid': True,
            'errors': []
        }

        if len(x_array) != len(y_array):
            validation_result['is_valid'] = False
            validation_result['errors'].append("X и Y должны иметь одинаковую длину")

        if not (np.issubdtype(x_array.dtype, np.number) and np.issubdtype(y_array.dtype, np.number)):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Данные должны быть числовыми")

        if np.isnan(x_array).any() or np.isnan(y_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют NaN значения")

        if np.isinf(x_array).any() or np.isinf(y_array).any():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Присутствуют бесконечные значения")

        if plot_type in ['bar', 'line']:
            if not np.all(np.diff(x_array) > 0):
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"Для {plot_type} рекомендуется упорядоченность X"
                )

        return validation_result

    @staticmethod
    def validate_3d_data(x: Optional[Union[List, np.ndarray, pd.Series]] = None,
                         y: Optional[Union[List, np.ndarray, pd.Series]] = None,
                         z: Union[List, np.ndarray, pd.Series] = None,
                         plot_type: str = 'scatter3d') -> dict:
        validation_result = {
            'is_valid': True,
            'errors': []
        }

        if plot_type in ['scatter3d']:
            x_array = np.array(x, dtype=float)
            y_array = np.array(y, dtype=float)
            z_array = np.array(z, dtype=float)

            if len(set(map(len, [x_array, y_array, z_array]))) > 1:
                validation_result['is_valid'] = False
                validation_result['errors'].append("X, Y и Z должны иметь одинаковую длину")

            if not (np.issubdtype(x_array.dtype, np.number) and
                    np.issubdtype(y_array.dtype, np.number) and
                    np.issubdtype(z_array.dtype, np.number)):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Данные должны быть числовыми")

            if np.isnan(x_array).any() or np.isnan(y_array).any() or np.isnan(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют NaN значения")

            if np.isinf(x_array).any() or np.isinf(y_array).any() or np.isinf(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют бесконечные значения")

        elif plot_type in ['surface', 'contour', 'heatmap']:
            z_array = np.array(z)
            if z_array.ndim != 2:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Для surface/contour/heatmap требуется 2D массив")

            if not np.issubdtype(z_array.dtype, np.number):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Данные должны быть числовыми")

            if np.isnan(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют NaN значения")

            if np.isinf(z_array).any():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Присутствуют бесконечные значения")

        return validation_result


class Visualizer:
    def __init__(self, visualization_type: str):
        self.visualization_type = visualization_type
        self.x: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.y: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.z: Optional[Union[List, pd.Series, np.ndarray]] = None
        self.validation_result: Optional[dict] = None
        self.plot_params = {}  # Хранение дополнительных параметров для построения графика

    def load_data(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z
        return self

    def set_params(self, **kwargs):
        """
        Установка дополнительных параметров для построения графика.
        Можно передать любые параметры, поддерживаемые plotly.express или go.
        """
        self.plot_params.update(kwargs)
        return self

    def validate(self) -> bool:
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
        if not self.validate():
            error_message = "\n".join(self.validation_result['errors'])
            raise ValueError(f"Ошибка валидации данных:\n{error_message}")

        if self.visualization_type in ['histogram', 'kde', 'distribution']:
            return self._one_dimensional_plot(title)
        elif self.visualization_type in ['scatter', 'line', 'bar', 'area', 'box', 'violin']:
            return self._two_dimensional_plot(title)
        elif self.visualization_type in ['scatter3d', 'surface', 'contour', 'heatmap']:
            return self._three_dimensional_plot(title)

    def _one_dimensional_plot(self, title):
        if self.visualization_type == 'histogram':
            fig = px.histogram(x=self.x, title=title, **self.plot_params)
        elif self.visualization_type == 'kde':
            fig = px.density_contour(x=self.x, title=title, **self.plot_params)
        elif self.visualization_type == 'distribution':
            fig = px.ecdf(x=self.x, title=title, **self.plot_params)
        else:
            raise ValueError("Некорректный тип графика")
        return fig

    def _two_dimensional_plot(self, title):
        if self.visualization_type == 'scatter':
            fig = px.scatter(x=self.x, y=self.y, title=title, **self.plot_params)
        elif self.visualization_type == 'line':
            fig = px.line(x=self.x, y=self.y, title=title, **self.plot_params)
        elif self.visualization_type == 'bar':
            fig = px.bar(x=self.x, y=self.y, title=title, **self.plot_params)
        elif self.visualization_type == 'area':
            fig = px.area(x=self.x, y=self.y, title=title, **self.plot_params)
        elif self.visualization_type == 'box':
            fig = px.box(x=self.x, y=self.y, title=title, **self.plot_params)
        elif self.visualization_type == 'violin':
            fig = px.violin(x=self.x, y=self.y, title=title, **self.plot_params)
        else:
            raise ValueError("Некорректный тип графика")
        return fig

    def _three_dimensional_plot(self, title):
        if self.visualization_type == 'scatter3d':
            if isinstance(self.x, np.ndarray) and self.x.ndim > 1:
                x = self.x.ravel()
                y = self.y.ravel()
                z = self.z.ravel()
            else:
                x, y, z = self.x, self.y, self.z
            fig = px.scatter_3d(x=x, y=y, z=z, title=title, **self.plot_params)
        elif self.visualization_type == 'surface':
            # Для surface необходимо использовать go.Surface
            # Параметры, не поддерживаемые напрямую go.Surface, можно задавать через layout
            fig = go.Figure(
                data=[go.Surface(z=self.z, **{k: v for k, v in self.plot_params.items() if k not in ['title']})])
            fig.update_layout(title=title, **{k: v for k, v in self.plot_params.items() if k not in ['z']})
        elif self.visualization_type == 'contour':
            fig = go.Figure(
                data=go.Contour(z=self.z, **{k: v for k, v in self.plot_params.items() if k not in ['title']}))
            fig.update_layout(title=title, **{k: v for k, v in self.plot_params.items() if k not in ['z']})
        elif self.visualization_type == 'heatmap':
            fig = px.imshow(self.z, title=title, **self.plot_params)
        else:
            raise ValueError("Некорректный тип графика")
        return fig

    def get_figure(self, title: str = 'Visualization'):
        return self.plot(title=title)

    def show(self):
        fig = self.plot()
        fig.show()

    def save(self, filename='visualization.html'):
        fig = self.plot()
        fig.write_html(filename)
