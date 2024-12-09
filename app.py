import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import base64
import io
import numpy as np
import datetime
from dash.exceptions import PreventUpdate
from tools import Cleaner, Normalizer
from visualizer import Visualizer

# Предполагается, что вы импортируете ваши классы отдельно:
# from visualizer import Visualizer
# from tools import Cleaner, Normalizer

app = dash.Dash(__name__, suppress_callback_exceptions=True, prevent_initial_callbacks=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    html.Div("DataInsight - Анализ и визуализация данных",
             style={'fontSize': 24, 'fontWeight': 'bold', 'marginBottom': 20}),

    # Добавляем Store для хранения данных
    dcc.Store(id='stored-data-original'),  # Для исходных данных
    dcc.Store(id='stored-data-filtered'),  # Для фильтрованных данных
    dcc.Loading(id="loading", children=[
        dcc.Dropdown(id='x-column'),
        dcc.Dropdown(id='y-column'),
    ]),

    html.Div(id='page-content'),

])

main_page_layout = html.Div([
    html.H3("Загрузка набора данных"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Перетащите или выберите файл']),
        style={
            'width': '60%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Button("Перейти к просмотру", id='go-to-data', n_clicks=0, disabled=True)
])

data_page_layout = html.Div([
    html.H3("Просмотр и подготовка данных"),
    html.Div([
        html.Div([
            html.H4("Фильтры"),
            html.Div("Столбец:"),
            dcc.Dropdown(id='filter-column', placeholder='Выберите столбец'),
            html.Div("Оператор:"),
            dcc.Dropdown(
                id='filter-operator',
                options=[
                    {'label': '>', 'value': '>'},
                    {'label': '<', 'value': '<'},
                    {'label': '==', 'value': '=='},
                    {'label': '!=', 'value': '!='},
                    {'label': '>=', 'value': '>='},
                    {'label': '<=', 'value': '<='}
                ],
                placeholder='Выберите оператор'
            ),
            html.Div("Значение:"),
            dcc.Input(id='filter-value', placeholder='Введите значение', style={'width': '100%'}),
            html.Button("Применить фильтр", id='apply-filter', n_clicks=0),
            html.Br(), html.Br(),

            html.Br(), html.Br(),

            html.H4("Очистка данных"),
            html.Div("Удаление дубликатов:"),
            dcc.Dropdown(id='duplicate-column', placeholder='Столбец (необязательно)'),
            dcc.Dropdown(
                id='duplicate-keep',
                options=[
                    {'label': 'Первый', 'value': 'first'},
                    {'label': 'Последний', 'value': 'last'},
                    {'label': 'Все', 'value': False}
                ],
                placeholder='Способ сохранения дубликатов'
            ),
            html.Button("Удалить дубликаты", id='clean-duplicates', n_clicks=0),
            html.Br(), html.Br(),

            html.H4("Нормализация данных"),
            dcc.Checklist(id='normalize-columns', inline=True),
            html.Button("Min-Max нормализация", id='minmax-normalize', n_clicks=0),
            html.Button("Z-score нормализация", id='zscore-normalize', n_clicks=0),
            html.Br(), html.Br(),

            html.Button("Перейти к визуализации", id='go-to-viz', n_clicks=0)
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),

        html.Div([
            html.H4("Предпросмотр данных"),
            dash_table.DataTable(
                id='data-table',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
])

viz_page_layout = html.Div([
    html.H3("Визуализация данных"),
    html.Div([
        html.Div("Тип визуализации:"),
        dcc.Dropdown(
            id='viz-type',
            options=[
                {'label': 'Линейный график', 'value': 'line'},
                {'label': 'Столбчатая диаграмма', 'value': 'bar'},
                {'label': 'Точечная диаграмма', 'value': 'scatter'}
            ],
            value='line'
        ),
        html.Div("Столбец X:"),
        dcc.Dropdown(id='x-column'),
        html.Div("Столбец Y:"),
        dcc.Dropdown(id='y-column'),
        html.Button("Построить график", id='build-plot', n_clicks=0)
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '20px'}),
    html.Div([
        dcc.Graph(id='main-graph')
    ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'})
])


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/view-data':
        return data_page_layout
    elif pathname == '/viz':
        return viz_page_layout
    else:
        return main_page_layout


@app.callback(
    Output('go-to-data', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def enable_go_to_data(contents, filename):
    if contents is not None and filename:
        return False
    return True


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename.lower():
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
        return pd.read_excel(io.BytesIO(decoded))
    else:
        return pd.DataFrame()


@app.callback(
    Output('stored-data-original', 'data'),
    Output('stored-data-filtered', 'data', allow_duplicate=True),
    Output('url', 'pathname', allow_duplicate=True),
    Input('go-to-data', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(n_clicks, contents, filename):
    if n_clicks > 0 and contents is not None and filename is not None:
        df = parse_contents(contents, filename)
        if not df.empty:
            return df.to_json(date_format='iso', orient='split'), df.to_json(date_format='iso',
                                                                             orient='split'), '/view-data'
    raise PreventUpdate


@app.callback(
    Output('data-table', 'columns'),
    Output('filter-column', 'options'),
    Output('duplicate-column', 'options'),
    Output('normalize-columns', 'options'),
    Input('stored-data-filtered', 'data')
)
def update_table(filtered_data_json):
    if filtered_data_json:
        df_filtered = pd.read_json(io.StringIO(filtered_data_json), orient='split')

        # Конвертация типов
        for c in df_filtered.columns:
            if df_filtered[c].dtype == object:
                try:
                    df_filtered[c] = pd.to_datetime(df_filtered[c], infer_datetime_format=True, errors='coerce')
                except Exception as e:
                    print(f"Error converting column {c} to datetime: {e}")

        columns_info = [{'name': f"{col} ({df_filtered[col].dtype})", 'id': col} for col in df_filtered.columns]
        filter_options = [{'label': c, 'value': c} for c in df_filtered.columns]
        duplicate_options = [{'label': c, 'value': c} for c in df_filtered.columns]
        normalize_options = [{'label': c, 'value': c} for c in df_filtered.columns if
                             pd.api.types.is_numeric_dtype(df_filtered[c])]

        return columns_info, filter_options, duplicate_options, normalize_options
    return [], [], [], [], []


@app.callback(
    Output('stored-data-filtered', 'data', allow_duplicate=True),
    Output('data-table', 'data'),
    Input('apply-filter', 'n_clicks'),
    Input('clean-duplicates', 'n_clicks'),
    Input('minmax-normalize', 'n_clicks'),
    Input('zscore-normalize', 'n_clicks'),
    State('stored-data-original', 'data'),
    State('stored-data-filtered', 'data'),
    State('filter-column', 'value'),
    State('filter-operator', 'value'),
    State('filter-value', 'value'),
    State('duplicate-column', 'value'),
    State('duplicate-keep', 'value'),
    State('normalize-columns', 'value')
)
def update_data_table(
        filter_clicks, dup_clicks, minmax_clicks, zscore_clicks,
        original_data_json, filtered_data_json,
        filter_column, filter_op, filter_val,
        dup_col, dup_keep,
        norm_columns
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if original_data_json is None:
        raise PreventUpdate

    df_original = pd.read_json(io.StringIO(original_data_json), orient='split')
    df_filtered = pd.read_json(io.StringIO(filtered_data_json),
                               orient='split') if filtered_data_json else df_original.copy()

    cleaner = Cleaner()
    normalizer = Normalizer()

    if triggered_id == 'apply-filter':
        if filter_column and filter_op and filter_val is not None:
            try:
                # Попытка преобразовать значение к числу
                val_parsed = float(filter_val)
            except ValueError:
                val_parsed = filter_val

            try:
                if filter_op == '==':
                    df_filtered = df_filtered[df_filtered[filter_column] == val_parsed]
                elif filter_op == '!=':
                    df_filtered = df_filtered[df_filtered[filter_column] != val_parsed]
                elif filter_op == '>':
                    df_filtered = df_filtered[df_filtered[filter_column] > val_parsed]
                elif filter_op == '<':
                    df_filtered = df_filtered[df_filtered[filter_column] < val_parsed]
                elif filter_op == '>=':
                    df_filtered = df_filtered[df_filtered[filter_column] >= val_parsed]
                elif filter_op == '<=':
                    df_filtered = df_filtered[df_filtered[filter_column] <= val_parsed]
            except:
                # В случае ошибки возвращаем исходные данные
                df_filtered = df_original.copy()

    elif triggered_id == 'clean-duplicates':
        if not df_filtered.empty:
            cleaner.load_data(df_filtered)
            cleaner.clean_duplicates(dup_col, keep=dup_keep)
            df_filtered = cleaner.data.copy()

    elif triggered_id == 'minmax-normalize':
        if norm_columns and not df_filtered.empty:
            normalizer.load_data(df_filtered)
            normalizer.min_max_normalize(norm_columns)
            df_filtered = normalizer.data.copy()

    elif triggered_id == 'zscore-normalize':
        if norm_columns and not df_filtered.empty:
            normalizer.load_data(df_filtered)
            normalizer.z_score_normalize(norm_columns)
            df_filtered = normalizer.data.copy()

    # Обновляем stored-data-filtered
    filtered_data_updated = df_filtered.to_json(date_format='iso', orient='split')

    # Возвращаем обновленные данные для Store и таблицы
    return filtered_data_updated, df_filtered.head(50).to_dict('records')


@app.callback(
    Output('url', 'pathname'),
    Input('go-to-viz', 'n_clicks')
)
def go_to_viz_page(n):
    if n > 0:
        return '/viz'
    return dash.no_update


@app.callback(
    Output('x-column', 'options'),
    Output('y-column', 'options'),
    Input('url', 'pathname'),
    State('stored-data-filtered', 'data'),
)
def update_viz_options(pathname, filtered_data_json):
    if pathname != '/viz':
        raise PreventUpdate
    if filtered_data_json:
        try:
            df_filtered = pd.read_json(io.StringIO(filtered_data_json), orient='split')
            print("update_viz_options called. Columns:", df_filtered.columns.tolist())  # Отладочный вывод
            options = [{'label': c, 'value': c} for c in df_filtered.columns]
            return options, options
        except Exception as e:
            print(f"Error in update_viz_options: {e}")  # Отладочный вывод
            return [], []
    print("update_viz_options called but no data available.")  # Отладочный вывод
    return [], []


@app.callback(
    Output('main-graph', 'figure'),
    Input('build-plot', 'n_clicks'),
    State('viz-type', 'value'),
    State('x-column', 'value'),
    State('y-column', 'value'),
    State('stored-data-filtered', 'data')
)
def build_graph(n_clicks, viz_type, x_col, y_col, filtered_data_json):
    if n_clicks > 0 and x_col and y_col and filtered_data_json:
        df_filtered = pd.read_json(io.StringIO(filtered_data_json), orient='split')
        vis = Visualizer(visualization_type=viz_type)
        vis.load_data(df_filtered[x_col], df_filtered[y_col])
        fig = vis.get_figure(title="Результат визуализации")
        return fig
    return px.line()


if __name__ == '__main__':
    app.run_server(debug=True)
