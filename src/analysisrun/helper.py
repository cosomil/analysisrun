from typing import Dict, TypeGuard

import pandas as pd


def read_dict(filename: str, key: str, value: str) -> Dict[str, str]:
    """
    CSVファイルを読み込み、指定したカラムをキーと値にして辞書を作成する。

    Parameters
    ----------
    filename
        読み込むCSVファイルのパス
    key
        辞書のキーとなるカラム名
    value
        辞書の値となるカラム名
    """

    df = pd.read_csv(filename).astype(str)
    return dict(zip(df[key], df[value]))


def is_float(x: float | None) -> TypeGuard[float]:
    return x is not None
