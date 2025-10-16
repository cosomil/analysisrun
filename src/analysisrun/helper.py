from typing import Dict, TypeGuard

import pandas as pd
from pandas._typing import ReadCsvBuffer, FilePath


def read_dict(
    filepath_or_buffer: FilePath | ReadCsvBuffer[str] | ReadCsvBuffer[bytes],
    key: str,
    value: str,
) -> Dict[str, str]:
    """
    CSVファイルを読み込み、指定したカラムをキーと値にして辞書を作成する。

    Parameters
    ----------
    filepath_or_buffer
        読み込むCSVファイル
    key
        辞書のキーとなるカラム名
    value
        辞書の値となるカラム名
    """

    df = pd.read_csv(filepath_or_buffer).astype(str)
    return dict(zip(df[key], df[value]))


def is_float(x: float | None) -> TypeGuard[float]:
    return x is not None
