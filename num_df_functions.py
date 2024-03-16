from pandas import DataFrame


def get_mean(df: DataFrame, col: str | int) -> int | Exception:
    try:
        mn = df[col].mean()
        return mn

    except Exception as err:
        return err


def get_median(df: DataFrame, col: str | int) -> int | Exception:
    df: DataFrame
    try:
        md = df[col].median()
        return md

    except Exception as err:
        return err


def get_std(df: DataFrame, col: str | int) -> int | Exception:
    df: DataFrame
    try:
        std = df[col].std()
        return std

    except Exception as err:
        return err


def get_quantile(df: DataFrame, col: str | int, q) -> int | Exception:
    df: DataFrame
    try:
        qn = df[col].quantile(q)
        return qn

    except Exception as err:
        return err


def get_min(df: DataFrame, col: str | int) -> int | Exception:
    try:
        mi = df[col].min()
        return mi

    except Exception as err:
        return err


def get_max(df: DataFrame, col: str | int) -> int | Exception:
    try:
        mx = df[col].max()
        return mx

    except Exception as err:
        return err
