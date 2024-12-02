import plotly.express as px
import plotly.graph_objects as go
import logging
from functools import lru_cache
import pandas as pd
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVisualizer:
    def __init__(self, data, plot_type='line', x=None, y=None, width=None, height=None, size=None, title='', **kwargs):
        self.data = data
        self.plot_type = plot_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size
        self.title = title
        self.kwargs = kwargs

        self.validate_input()

    def validate_input(self):
        if self.data is None or len(self.data) == 0:
            raise ValueError("Data can't be empty.")

        supported_plot_types = (
            'line', 'scatter', 'histogram', 'bar', 'box', 'violin',
            'heatmap', 'pie', 'sunburst', 'treemap', 'scatter_geo',
            'image', 'candlestick'
        )

        if self.plot_type not in supported_plot_types:
            raise ValueError(f"Unsupported chart type: {self.plot_type}")

    @lru_cache(maxsize=32)
    def create_plot(self):
        logger.info(f"Creating a chart type '{self.plot_type}' with parameters: {self.kwargs}")

        if self.plot_type == 'line':
            fig = px.line(self.data, x=self.x, y=self.y, width=self.width, height=self.height,
                          title=self.title, **self.kwargs)

        elif self.plot_type == 'scatter':
            fig = px.scatter(self.data, x=self.x, y=self.y, width=self.width, height=self.height,
                             size=self.size, title=self.title, **self.kwargs)

        elif self.plot_type == 'histogram':
            fig = px.histogram(self.data, x=self.x, width=self.width, height=self.height, title=self.title,
                               **self.kwargs)

        elif self.plot_type == 'bar':
            fig = px.bar(self.data, x=self.x, y=self.y, width=self.width, height=self.height, title=self.title,
                         **self.kwargs)

        elif self.plot_type == 'box':
            fig = px.box(self.data, x=self.x, y=self.y, width=self.width, height=self.height, title=self.title,
                         **self.kwargs)

        elif self.plot_type == 'violin':
            fig = px.violin(self.data, x=self.x, y=self.y, width=self.width, height=self.height, title=self.title,
                            **self.kwargs)

        elif self.plot_type == 'heatmap':
            fig = px.imshow(self.data, width=self.width, height=self.height, title=self.title, **self.kwargs)

        elif self.plot_type == 'pie':
            fig = px.pie(self.data, names=self.x, values=self.y, width=self.width, height=self.height, title=self.title,
                         **self.kwargs)

        elif self.plot_type == 'sunburst':
            fig = px.sunburst(self.data, path=self.x, width=self.width, height=self.height, values=self.y,
                              title=self.title, **self.kwargs)

        elif self.plot_type == 'treemap':
            fig = px.treemap(self.data, path=self.x, width=self.width, height=self.height, values=self.y,
                             title=self.title, **self.kwargs)

        elif self.plot_type == 'scatter_geo':
            fig = px.scatter_geo(self.data, lat=self.kwargs.pop('lat'), lon=self.kwargs.pop('lon'), width=self.width,
                                 height=self.height, size=self.size, title=self.title, **self.kwargs)

        elif self.plot_type == 'image':
            fig = px.imshow(self.data, width=self.width, height=self.height, title=self.title, **self.kwargs)

        elif self.plot_type == 'candlestick':
            fig = go.Figure(data=[go.Candlestick(
                x=self.data[self.x],
                open=self.data[self.kwargs.pop('open', None)],
                high=self.data[self.kwargs.pop('high', None)],
                low=self.data[self.kwargs.pop('low', None)],
                close=self.data[self.kwargs.pop('close', None)],
            )])

            fig.update_layout(title=self.title, width=self.width, height=self.height)
        else:
            raise ValueError(f"Unsupported chart type: {self.plot_type}")

        fig.update_layout(
            template=self.kwargs.pop('template', 'plotly'),
            font=dict(
                family=self.kwargs.pop('font_family', 'Arial'),
                size=self.kwargs.pop('font_size', 12),
                color=self.kwargs.pop('font_color', '#000000')
            ),
            colorway=self.kwargs.pop('colorway', None),
            xaxis=dict(type='date')
        )

        # Добавление интерактивных элементов
        if 'hover_data' in self.kwargs:
            fig.update_traces(hoverinfo=self.kwargs['hover_data'])
        if 'annotations' in self.kwargs:
            fig.update_layout(annotations=self.kwargs['annotations'])

        return fig

    def update_data(self, new_data):
        logger.info("Updating data. New columns: %s", ", ".join(self.data.columns.tolist()))

        self.data = new_data

    def update_kwargs(self, **new_kwargs):
        logger.info("Updating kwargs. New kwargs: %s", ", ".join(f"{key}={value}" for key, value in new_kwargs.items()))

        self.kwargs.update(new_kwargs)

    def update_resolution(self, new_width, new_height):
        logger.info(f"Updating resolution. New (width, height): ({new_width}, {new_height})")

        self.width = new_width
        self.height = new_height

    def get_dash_component(self):
        from dash import dcc
        fig = self.create_plot()
        return dcc.Graph(figure=fig)

    def save_plot(self, file_path, file_format='html'):
        logger.info(f"Saving plot as {file_format} to {file_path}")

        fig = self.create_plot()
        if file_format == 'html':
            fig.write_html(file_path)

        elif file_format in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            fig.write_image(file_path)

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    async def create_plot_async(self):
        loop = asyncio.get_event_loop()
        fig = await loop.run_in_executor(None, self.create_plot)
        return fig

    def aggregate_data(self, group_by, agg_func='mean'):
        self.data = self.data.groupby(group_by).agg(agg_func).reset_index()

    def load_data_from_db(self, query, connection):
        self.data = pd.read_sql(query, connection)

    def parse_dates(self, date_column):
        self.data[date_column] = pd.to_datetime(self.data[date_column])
