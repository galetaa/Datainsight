import pandas as pd
from pandas import DataFrame


def get_df_from_file(file_name: str, **kwargs) -> DataFrame | list[DataFrame] | Exception:
    df: DataFrame
    try:
        extension: str = file_name[file_name.rindex('.'):]

        if extension == ".csv":
            df = pd.read_csv(file_name, **kwargs)

        elif extension == ".xlsx":
            df = pd.read_excel(file_name, **kwargs)

        elif extension == ".json":
            df = pd.read_json(file_name, **kwargs)

        elif extension == ".html":
            df: list[DataFrame] = pd.read_html(file_name, **kwargs)


        else:
            df: DataFrame = DataFrame()

        return df

    except Exception as err:
        return err
