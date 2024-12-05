import json
from collections import Counter

import pandas as pd


def get_delimiter(file_name, max_sample_lines=10):
    """
    Detect the delimiter used in an SV (separator value) file.

    Parameters:
    file_name (str): Path to the SV file.
    max_sample_lines (int): The maximum number of lines to sample for delimiter detection.

    Returns:
    str: The detected delimiter.
    """

    with open(file_name, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines(max_sample_lines)
        delimiters = [',', '\t', ';', '|', ' ']
        delimiter_counts = Counter()

        for line in lines:
            for delimiter in delimiters:
                delimiter_counts[delimiter] += line.count(delimiter)

    most_common_delimiter = delimiter_counts.most_common(1)[0][0]
    return most_common_delimiter


def are_excel_headers_vertical(file_path, rows_to_read=1000, cols_to_read=1000):
    """
    Определяет, расположены ли заголовки вертикально в части файла Excel.

    Parameters:
    file_path (str): Путь к файлу Excel (xlsx или xls).
    rows_to_read (int): Количество строк для чтения и анализа.
    cols_to_read (int): Количество столбцов для чтения и анализа.

    Returns:
    bool: True, если заголовки расположены вертикально, иначе False.
    """
    # Чтение указанной части файла без предполагаемых заголовков
    df = pd.read_excel(file_path, header=None, nrows=rows_to_read, usecols=list(range(cols_to_read)))

    # Проверка формата данных в первом столбце и первой строке
    first_col_dtypes = df.dtypes[0]
    first_row_dtypes = df.iloc[0].apply(type)

    # Определение, состоят ли все значения в одной из областей из одного типа (например, строк)
    vertical_headers = first_col_dtypes == object and all(isinstance(x, type(first_col_dtypes)) for x in df.iloc[:, 0])
    horizontal_headers = first_row_dtypes == object and all(
        isinstance(x, type(first_row_dtypes[0])) for x in df.iloc[0, :])

    # Возвращаем True, если заголовки вероятно вертикальные
    return vertical_headers and not horizontal_headers


def determine_json_orientation_from_file(file_path, lines_to_read=10):
    """
    Определяет ориентацию части файла JSON.

    Parameters:
    file_path (str): Путь к файлу JSON.
    lines_to_read (int): Количество строк, которые будут прочитаны для анализа.

    Returns:
    str: Ориентация JSON ('split', 'records', 'index', 'unknown', 'invalid JSON').
    """
    try:
        with open(file_path, 'r') as file:
            # Чтение заданного количества строк из файла
            lines = ''.join([next(file) for _ in range(lines_to_read)])
            data = json.loads(lines)

            # Проверка ориентации
            if all(key in data for key in ['columns', 'index', 'data']):
                return 'split'
            if isinstance(data, list) and all(isinstance(record, dict) for record in data):
                return 'records'
            if isinstance(data, dict) and all(isinstance(data[key], list) for key in data.keys()):
                return 'index'

    except json.JSONDecodeError:
        return 'invalid JSON'
    except StopIteration:
        return 'invalid JSON or insufficient data'

    return 'unknown'


print(are_excel_headers_vertical("gg.xlsx"))
