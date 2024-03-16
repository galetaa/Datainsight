from pandas import DataFrame


def find_missing_values_(df) -> dict[DataFrame] | Exception:
    try:
        missing_values: dict = {}
        for column in df.columns:
            missing_rows = df[df[column].isnull()]
            if not missing_rows.empty:
                missing_values[column] = missing_rows

        return missing_values

    except Exception as err:
        return err
