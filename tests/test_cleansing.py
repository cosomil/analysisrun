"""
cleansing.pyのテストコード
"""

import pandas as pd

from analysisrun.cleansing import CleansedData, filter_by_entity


def test_filter_by_entity_with_dataframe():
    """pandas.DataFrameを入力とした場合のfilter_by_entity関数のテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Nuclei", "Activity Spots", "Cytoplasm"],
            "Value": [10, 20, 30, 40],
            "Id": [1, 2, 3, 4],
        }
    )

    # デフォルトのエンティティ（"Activity Spots"）でフィルタリング
    result = filter_by_entity(data)

    # 結果の検証
    assert isinstance(result, CleansedData)
    assert len(result._data) == 2
    assert all(result._data["Entity"] == "Activity Spots")
    assert list(result._data["Value"]) == [10, 30]
    assert list(result._data["Id"]) == [1, 3]


def test_filter_by_entity_with_custom_entity():
    """カスタムエンティティ名を指定した場合のテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Nuclei", "Activity Spots", "Nuclei"],
            "Value": [10, 20, 30, 40],
            "Id": [1, 2, 3, 4],
        }
    )

    # "Nuclei"エンティティでフィルタリング
    result = filter_by_entity(data, entity="Nuclei")

    # 結果の検証
    assert isinstance(result, CleansedData)
    assert len(result._data) == 2
    assert all(result._data["Entity"] == "Nuclei")
    assert list(result._data["Value"]) == [20, 40]
    assert list(result._data["Id"]) == [2, 4]


def test_filter_by_entity_with_cleansed_data():
    """CleansedDataを入力とした場合のテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Nuclei", "Activity Spots", "Cytoplasm"],
            "Value": [10, 20, 30, 40],
            "Id": [1, 2, 3, 4],
        }
    )

    # 既にクレンジング済みのデータを作成
    cleansed_data = CleansedData(_data=data)

    # フィルタリング実行
    result = filter_by_entity(cleansed_data)

    # 結果の検証
    assert isinstance(result, CleansedData)
    assert len(result._data) == 2
    assert all(result._data["Entity"] == "Activity Spots")
    assert list(result._data["Value"]) == [10, 30]
    assert list(result._data["Id"]) == [1, 3]


def test_filter_by_entity_no_matches():
    """指定したエンティティに一致するデータが存在しない場合のテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {"Entity": ["Nuclei", "Cytoplasm"], "Value": [20, 40], "Id": [2, 4]}
    )

    # 存在しないエンティティでフィルタリング
    result = filter_by_entity(data, entity="Activity Spots")

    # 結果の検証（空のDataFrameが返される）
    assert isinstance(result, CleansedData)
    assert len(result._data) == 0
    assert result._data.empty


def test_filter_by_entity_empty_dataframe():
    """空のDataFrameを入力とした場合のテスト"""
    # 空のテストデータの作成
    data = pd.DataFrame(columns=["Entity", "Value", "Id"])

    # フィルタリング実行
    result = filter_by_entity(data)

    # 結果の検証
    assert isinstance(result, CleansedData)
    assert len(result._data) == 0
    assert result._data.empty


def test_filter_by_entity_case_sensitive():
    """エンティティ名の大文字小文字が区別されることのテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {
            "Entity": ["activity spots", "Activity Spots", "ACTIVITY SPOTS"],
            "Value": [10, 20, 30],
            "Id": [1, 2, 3],
        }
    )

    # デフォルトのエンティティでフィルタリング
    result = filter_by_entity(data)

    # 結果の検証（完全一致のみ）
    assert isinstance(result, CleansedData)
    assert len(result._data) == 1
    assert all(result._data["Entity"] == "Activity Spots")
    assert list(result._data["Value"]) == [20]
    assert list(result._data["Id"]) == [2]


def test_filter_by_entity_preserves_dataframe_structure():
    """フィルタリング後もDataFrameの構造が保持されることのテスト"""
    # 複数の列を持つテストデータの作成
    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Nuclei", "Activity Spots"],
            "Value": [10, 20, 30],
            "Id": [1, 2, 3],
            "Name": ["A", "B", "C"],
            "Score": [0.1, 0.2, 0.3],
            "Index": [100, 200, 300],
        }
    )

    # フィルタリング実行
    result = filter_by_entity(data)

    # 結果の検証
    assert isinstance(result, CleansedData)
    assert len(result._data) == 2
    assert list(result._data.columns) == [
        "Entity",
        "Value",
        "Id",
        "Name",
        "Score",
        "Index",
    ]
    assert all(result._data["Entity"] == "Activity Spots")
    assert list(result._data["Value"]) == [10, 30]
    assert list(result._data["Name"]) == ["A", "C"]
    assert list(result._data["Score"]) == [0.1, 0.3]
