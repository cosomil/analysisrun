from typing import Dict

import pandas as pd


def read_dict(filename: str, key: str, value: str) -> Dict[str, str]:
    """
    CSVファイルを読み込み、指定したカラムをキーと値にして辞書を作成する。

    Parameters
    ----------
    filename : str
        読み込むCSVファイルのパス
    key : str
        辞書のキーとなるカラム名
    value : str
        辞書の値となるカラム名
    """

    df = pd.read_csv(filename).astype(str)
    return dict(zip(df[key], df[value]))
