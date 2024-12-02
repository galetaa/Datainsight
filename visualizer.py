import plotly.express as px
import plotly.graph_objects as go
import logging
from functools import lru_cache
import pandas as pd
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVisualizer:
    def __init__(self, data, plot_type='line', x=None, y=None, width=None, height=None, color=None, size=None, title='',
                 **kwargs):
        self.data = data
        self.plot_type = plot_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.size = size
        self.title = title
        self.kwargs = kwargs
        self.validate_input()

    def validate_input(self):
        if self.data is None or len(self.data) == 0:
            raise ValueError("Данные не должны быть пустыми.")

        supported_plot_types = (
            'line', 'scatter', 'histogram', 'bar', 'box', 'violin',
            'heatmap', 'pie', 'sunburst', 'treemap', 'scatter_geo',
            'image', 'candlestick'
        )
        if self.plot_type not in supported_plot_types:
            raise ValueError(f"Неподдерживаемый тип графика: {self.plot_type}")

    @lru_cache(maxsize=32)
    def create_plot(self):
        logger.info(f"Создание графика типа '{self.plot_type}' с параметрами {self.kwargs}")

        if self.plot_type == 'line':
            fig = px.line(self.data, x=self.x, y=self.y, width=self.width, height=self.height, color=self.color,
                          title=self.title, **self.kwargs)

        elif self.plot_type == 'scatter':
            fig = px.scatter(self.data, x=self.x, y=self.y, width=self.width, height=self.height, color=self.color,
                             size=self.size, title=self.title,**self.kwargs)

        elif self.plot_type == 'histogram':
            fig = px.histogram(self.data, x=self.x, color=self.color, title=self.title, **self.kwargs)

        elif self.plot_type == 'bar':
            fig = px.bar(self.data, x=self.x, y=self.y, color=self.color, title=self.title, **self.kwargs)

        elif self.plot_type == 'box':
            fig = px.box(self.data, x=self.x, y=self.y, color=self.color, title=self.title, **self.kwargs)

        elif self.plot_type == 'violin':
            fig = px.violin(self.data, x=self.x, y=self.y, color=self.color, title=self.title, **self.kwargs)

        elif self.plot_type == 'heatmap':
            fig = px.imshow(self.data, title=self.title, **self.kwargs)

        elif self.plot_type == 'pie':
            fig = px.pie(self.data, names=self.x, values=self.y, title=self.title, **self.kwargs)

        elif self.plot_type == 'sunburst':
            fig = px.sunburst(self.data, path=self.x, values=self.y, title=self.title, **self.kwargs)

        elif self.plot_type == 'treemap':
            fig = px.treemap(self.data, path=self.x, values=self.y, title=self.title, **self.kwargs)

        elif self.plot_type == 'scatter_geo':
            fig = px.scatter_geo(self.data, lat=self.kwargs.pop('lat'), lon=self.kwargs.pop('lon'), color=self.color,
                                 size=self.size, title=self.title, **self.kwargs)

        elif self.plot_type == 'image':
            fig = px.imshow(self.data, title=self.title, **self.kwargs)

        elif self.plot_type == 'candlestick':
            fig = go.Figure(data=[go.Candlestick(
                x=self.data[self.x],
                open=self.data[self.kwargs.pop('open')],
                high=self.data[self.kwargs.pop('high')],
                low=self.data[self.kwargs.pop('low')],
                close=self.data[self.kwargs.pop('close')]
            )])
            fig.update_layout(title=self.title)
        else:
            raise ValueError(f"Неподдерживаемый тип графика: {self.plot_type}")

        # Настройка оформления графика
        fig.update_layout(
            template=self.kwargs.get('template', 'plotly'),
            font=dict(
                family=self.kwargs.get('font_family', 'Arial'),
                size=self.kwargs.get('font_size', 12),
                color=self.kwargs.get('font_color', '#000000')
            ),
            colorway=self.kwargs.get('colorway', None),
            xaxis=dict(type='date')
        )

        # Добавление интерактивных элементов
        if 'hover_data' in self.kwargs:
            fig.update_traces(hoverinfo=self.kwargs['hover_data'])
        if 'annotations' in self.kwargs:
            fig.update_layout(annotations=self.kwargs['annotations'])

        return fig

    def update_data(self, new_data):
        """Обновляет данные визуализатора."""
        self.data = new_data

    def update_kwargs(self, **kwargs):
        """Обновляет параметры графика."""
        self.kwargs.update(kwargs)

    def get_dash_component(self):
        """Возвращает Dash-компонент для отображения графика."""
        from dash import dcc
        fig = self.create_plot()
        return dcc.Graph(figure=fig)

    def save_plot(self, file_path, format='html'):
        """Сохраняет график в файл указанного формата."""
        fig = self.create_plot()
        if format == 'html':
            fig.write_html(file_path)
        elif format in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            fig.write_image(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {format}")

    async def create_plot_async(self):
        """Асинхронно создаёт и возвращает объект Figure с графиком."""
        loop = asyncio.get_event_loop()
        fig = await loop.run_in_executor(None, self.create_plot)
        return fig

    def aggregate_data(self, group_by, agg_func='mean'):
        """Агрегирует данные по указанным столбцам."""
        self.data = self.data.groupby(group_by).agg(agg_func).reset_index()

    def load_data_from_db(self, query, connection):
        """Загружает данные из базы данных."""
        self.data = pd.read_sql(query, connection)

    def parse_dates(self, date_column):
        """Преобразует столбец с датами в формат datetime."""
        self.data[date_column] = pd.to_datetime(self.data[date_column])
