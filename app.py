import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table, callback_context
import pandas as pd
import io,json
import base64

# Импорт ранее созданных классов
from data_viewer import DataViewer
from visualizer import Visualizer


class DataInsightApp:
    def __init__(self):
        self.app = dash.Dash(__name__, prevent_initial_callbacks=True,
                             external_stylesheets=[dbc.themes.BOOTSTRAP],
                             suppress_callback_exceptions=True)
        self.current_dataframe = None
        self.setup_layout()
        self.register_callbacks()

    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Заголовок приложения
            dbc.Row([
                dbc.Col(html.H1("DataInsight", className="text-center my-4"), width=12)
            ]),

            # Область загрузки данных
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Перетащите или ',
                            html.A('выберите файл')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status')
                ], width=12)
            ], className="mb-4"),

            # Вкладки для работы с данными
            dbc.Tabs([
                dbc.Tab(label="Редактирование данных", children=[
                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                id='editable-table',
                                columns=[],
                                data=[],
                                editable=True,
                                filter_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                column_selectable="single",
                                row_selectable="multi",
                                row_deletable=True,
                                page_action="native",
                                page_current=0,
                                page_size=10,
                            ),
                            html.Div([
                                dbc.Button("Сохранить изменения", id="save-changes-btn", color="primary"),
                                dbc.Button("Отменить изменения", id="reset-changes-btn", color="secondary")
                            ])
                        ])
                    ])
                ]),

                dbc.Tab(label="Анализ данных", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='analysis-type-dropdown',
                                options=[
                                    {'label': 'Общий анализ', 'value': 'overall'},
                                    {'label': 'Статистические тесты', 'value': 'statistical'},
                                    {'label': 'Корреляционный анализ', 'value': 'correlation'}
                                ],
                                placeholder="Выберите тип анализа"
                            ),
                            html.Div(id='analysis-results')
                        ])
                    ])
                ]),

                dbc.Tab(label="Визуализация", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='visualization-type-dropdown',
                                options=[
                                    {'label': 'Гистограмма', 'value': 'histogram'},
                                    {'label': 'Scatter', 'value': 'scatter'},
                                    {'label': 'Линейный', 'value': 'line'},
                                    {'label': '3D Scatter', 'value': 'scatter3d'},
                                    {'label': 'KDE', 'value': 'kde'},
                                    {'label': 'Распределение', 'value': 'distribution'},
                                    {'label': 'Столбчатый', 'value': 'bar'},
                                    {'label': 'Площадь', 'value': 'area'},
                                    {'label': 'Коробчатый', 'value': 'box'},
                                    {'label': 'Скрипичный', 'value': 'violin'},
                                    {'label': 'Поверхность', 'value': 'surface'},
                                    {'label': 'Контур', 'value': 'contour'},
                                    {'label': 'Тепловая карта', 'value': 'heatmap'}
                                ],
                                placeholder="Выберите тип графика"
                            ),
                            dcc.Dropdown(
                                id='x-axis-column',
                                placeholder="Выберите колонку для X",
                                style={'display': 'none'}
                            ),
                            dcc.Dropdown(
                                id='y-axis-column',
                                placeholder="Выберите колонку для Y",
                                style={'display': 'none'}
                            ),
                            dcc.Dropdown(
                                id='z-axis-column',
                                placeholder="Выберите колонку для Z",
                                style={'display': 'none'}
                            ),
                            dcc.Graph(id='visualization-output')
                        ])
                    ])
                ])
            ])
        ], fluid=True)

    def parse_contents(self, contents, filename):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return None

            return df
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return None

    def register_callbacks(self):
        @self.app.callback(
            [Output('editable-table', 'columns'),
             Output('editable-table', 'data')],
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename')]
        )
        def update_editable_table(contents, filename):
            if contents is not None:
                df = self.parse_contents(contents, filename)

                columns = [
                    {'name': i, 'id': i, 'deletable': True,
                     'renamable': True} for i in df.columns
                ]

                return columns, df.to_dict('records')

            return [], []

        @self.app.callback(
            [Output('upload-status', 'children'),
             Output('x-axis-column', 'options'),
             Output('y-axis-column', 'options'),
             Output('z-axis-column', 'options', allow_duplicate=True)],
            [Input('upload-data', 'contents'),
             Input('save-changes-btn', 'n_clicks')],
            [State('upload-data', 'filename'),
             State('editable-table', 'data')]
        )
        def update_data(contents, n_clicks, filename, edited_data):
            ctx = callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle file upload
            if triggered_id == 'upload-data' and contents is not None:
                df = self.parse_contents(contents, filename)

                if df is not None:
                    self.current_dataframe = df

                    # Превью данных
                    preview_table = dbc.Table.from_dataframe(
                        df.head(10),
                        striped=True,
                        bordered=True,
                        hover=True
                    )

                    # Опции для колонок визуализации
                    column_options = [{'label': col, 'value': col} for col in df.columns]

                    return (
                        f"Файл {filename} успешно загружен",
                        column_options,
                        column_options,
                        column_options
                    )

            # Handle save changes
            elif triggered_id == 'save-changes-btn' and n_clicks and edited_data:
                df_edited = pd.DataFrame(edited_data)
                self.current_dataframe = df_edited

                # Опции для колонок визуализации
                column_options = [{'label': col, 'value': col} for col in df_edited.columns]

                return (
                    "Данные успешно обновлены",
                    column_options,
                    column_options,
                    column_options
                )

            # Default return if no action is taken
            return "Загрузите файл", [], [], []

        @self.app.callback(
            Output('analysis-results', 'children'),
            [Input('analysis-type-dropdown', 'value')]
        )
        def perform_analysis(analysis_type):
            if self.current_dataframe is None:
                return "Сначала загрузите данные"

            viewer = DataViewer(self.current_dataframe)

            if analysis_type == 'overall':
                analysis = viewer.analyze_overall()
                return html.Div([
                    html.H3("Общий обзор данных", className='mb-4'),
                    html.Div([
                        html.P(f"Общее количество строк: {analysis['shape'][0]}"),
                        html.P(f"Общее количество столбцов: {analysis['shape'][1]}"),

                        html.H4("Типы данных в столбцах:", className='mt-3'),
                        html.Ul([
                            html.Li(f"{col}: {dtype}") for col, dtype in analysis['dtypes'].items()
                        ]),

                        html.H4("Пропуски в данных:", className='mt-3'),
                        html.Table([
                            html.Thead(html.Tr(
                                [html.Th("Столбец"), html.Th("Количество пропусков"), html.Th("Процент пропусков")])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(col),
                                    html.Td(f"{analysis['missing_count'].get(col, 0)}"),
                                    html.Td(f"{analysis['missing_percent'].get(col, 0):.2f}%")
                                ]) for col in analysis['missing_count']
                            ])
                        ], className='table table-striped'),

                        html.P(f"Количество дубликатов: {analysis['duplicates']}", className='mt-3 font-weight-bold')
                    ])
                ])

            elif analysis_type == 'statistical':
                tests = viewer.statistical_tests()
                return html.Div([
                    html.H3("Статистический анализ"),
                    html.Table([
                                   html.Tr([
                                       html.Th("Столбец"),
                                       html.Th("Тест Шапиро-Уилка"),
                                       html.Th("Асимметрия"),
                                       html.Th("Эксцесс")
                                   ])
                               ] + [
                                   html.Tr([
                                       html.Td(col),
                                       html.Td(
                                           f"{test['shapiro_test']['statistic']:.4f} ({'Нормальное' if test['shapiro_test']['is_normal_distribution'] else 'Не нормальное'})",
                                           style={'color': 'green' if test['shapiro_test'][
                                               'is_normal_distribution'] else 'red'}),
                                       html.Td(f"{test['skewness']:.4f}"),
                                       html.Td(f"{test['kurtosis']:.4f}")
                                   ]) for col, test in tests.items()
                               ], className='table table-striped')
                ])

            elif analysis_type == 'correlation':
                correlations = viewer.correlation_analysis()
                return html.Div([
                    html.H3("Корреляционный анализ"),
                    html.Div([
                        html.H4("Сильные корреляции"),
                        html.Table([
                                       html.Tr([html.Th("Столбцы"), html.Th("Значение корреляции"),
                                                html.Th("Тип корреляции")])
                                   ] + [
                                       html.Tr([
                                           html.Td(f"{corr['columns'][0]} - {corr['columns'][1]}"),
                                           html.Td(f"{corr['correlation']:.4f}"),
                                           html.Td(corr['correlation_type'],
                                                   style={'color': 'green' if corr[
                                                                                  'correlation_type'] == 'positive' else 'red'})
                                       ]) for corr in correlations['strong_correlations']
                                   ], className='table table-striped')
                    ])
                ])

        @self.app.callback(
            [Output('x-axis-column', 'style'),
             Output('y-axis-column', 'style'),
             Output('z-axis-column', 'style')],
            [Input('visualization-type-dropdown', 'value')]
        )
        def toggle_axis_columns(vis_type):
            # Default hidden state for all columns
            x_style = {'display': 'none'}
            y_style = {'display': 'none'}
            z_style = {'display': 'none'}

            # Mapping of visualization types to required axes
            axis_requirements = {
                'histogram': {'x': True},
                'kde': {'x': True},
                'distribution': {'x': True},
                'scatter': {'x': True, 'y': True},
                'line': {'x': True, 'y': True},
                'bar': {'x': True, 'y': True},
                'area': {'x': True, 'y': True},
                'box': {'x': True, 'y': True},
                'violin': {'x': True, 'y': True},
                'scatter3d': {'x': True, 'y': True, 'z': True},
                'surface': {'z': True},
                'contour': {'z': True},
                'heatmap': {'z': True}
            }

            # Show/hide columns based on visualization type
            if vis_type in axis_requirements:
                req = axis_requirements[vis_type]
                if req.get('x', False):
                    x_style = {'display': 'block'}
                if req.get('y', False):
                    y_style = {'display': 'block'}
                if req.get('z', False):
                    z_style = {'display': 'block'}

            return x_style, y_style, z_style

        @self.app.callback(
            Output('visualization-output', 'figure'),
            [Input('visualization-type-dropdown', 'value'),
             Input('x-axis-column', 'value'),
             Input('y-axis-column', 'value'),
             Input('z-axis-column', 'value')]
        )
        def update_visualization(vis_type, x_column, y_column, z_column):
            if self.current_dataframe is None or x_column is None:
                return {}

            visualizer = Visualizer(vis_type)

            # Добавляем поддержку всех типов визуализации из Visualizer
            visualization_mapping = {
                'histogram': lambda: (
                    visualizer.load_data(x=self.current_dataframe[x_column]).plot()
                    if x_column else {}
                ),
                'kde': lambda: (
                    visualizer.load_data(x=self.current_dataframe[x_column]).plot()
                    if x_column else {}
                ),
                'distribution': lambda: (
                    visualizer.load_data(x=self.current_dataframe[x_column]).plot()
                    if x_column else {}
                ),
                'scatter': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'line': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'bar': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'area': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'box': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'violin': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column]
                    ).plot()
                    if x_column and y_column else {}
                ),
                'scatter3d': lambda: (
                    visualizer.load_data(
                        x=self.current_dataframe[x_column],
                        y=self.current_dataframe[y_column],
                        z=self.current_dataframe[z_column]
                    ).plot()
                    if x_column and y_column and z_column else {}
                ),
                'surface': lambda: (
                    visualizer.load_data(z=self.current_dataframe[z_column]).plot()
                    if z_column else {}
                ),
                'contour': lambda: (
                    visualizer.load_data(z=self.current_dataframe[z_column]).plot()
                    if z_column else {}
                ),
                'heatmap': lambda: (
                    visualizer.load_data(z=self.current_dataframe[z_column]).plot()
                    if z_column else {}
                )
            }

            plot_func = visualization_mapping.get(vis_type)
            return plot_func() if plot_func else {}

    def run(self, debug=True):
        self.app.run_server(debug=debug)


# Запуск приложения
if __name__ == '__main__':
    app = DataInsightApp()
    app.run()
