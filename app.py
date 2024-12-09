import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table, callback_context
import pandas as pd
import io
import base64

# Импорт ранее созданных классов
from data_viewer import DataViewer
from tools import Cleaner
from tools import Normalizer
from visualizer import Visualizer


class DataInsightApp:
    def __init__(self):
        self.app = dash.Dash(__name__,prevent_initial_callbacks=True,
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
                dbc.Tab(label="Просмотр данных", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='dataset-preview')
                        ])
                    ])
                ]),
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
                                    {'label': '3D Scatter', 'value': 'scatter3d'}
                                ],
                                placeholder="Выберите тип графика"
                            ),
                            dcc.Dropdown(
                                id='x-axis-column',
                                placeholder="Выберите колонку для X"
                            ),
                            dcc.Dropdown(
                                id='y-axis-column',
                                placeholder="Выберите колонку для Y"
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
             Output('dataset-preview', 'children'),
             Output('x-axis-column', 'options'),
             Output('y-axis-column', 'options'),
             Output('z-axis-column', 'options',allow_duplicate=True)],
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
                        preview_table,
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
                    dbc.Table.from_dataframe(
                        df_edited.head(10),
                        striped=True,
                        bordered=True,
                        hover=True
                    ),
                    column_options,
                    column_options,
                    column_options
                )

            # Default return if no action is taken
            return "Загрузите файл", None, [], [], []

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
                return html.Pre(str(analysis))

            elif analysis_type == 'statistical':
                tests = viewer.statistical_tests()
                return html.Pre(str(tests))

            elif analysis_type == 'correlation':
                correlations = viewer.correlation_analysis()
                return html.Pre(str(correlations))

        @self.app.callback(
            [Output('z-axis-column', 'options',allow_duplicate=True),
             Output('z-axis-column', 'style',allow_duplicate=True)],
            [Input('visualization-type-dropdown', 'value')]
        )
        def toggle_z_axis(vis_type):
            if vis_type in ['scatter3d', 'surface', 'contour', 'heatmap']:
                return (
                    [{'label': col, 'value': col} for col in self.current_dataframe.columns],
                    {'display': 'block'}
                )
            return [], {'display': 'none'}

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
