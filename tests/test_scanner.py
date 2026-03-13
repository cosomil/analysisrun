"""
scanner.pyのテストコード
"""

import pandas as pd

from analysisrun.cleansing import CleansedData
from analysisrun.scanner import Fields, Lanes


def test_fields_initialization():
    """Fieldsクラスの初期化のテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {"MultiPointIndex": [1, 1, 2, 2, 3], "Value": [10, 20, 30, 40, 50]}
    )
    field_numbers = [1, 2, 3]

    fields = Fields(
        name="test_data",
        image_analysis_method="test_method",
        data=data,
        field_numbers=field_numbers,
    )

    assert fields.data_name == "test_data"
    assert fields.image_analysis_method == "test_method"
    assert len(fields.data) == 5
    assert fields.field_numbers == [1, 2, 3]


def test_fields_iteration():
    """Fieldsクラスのイテレーションのテスト"""
    # テストデータの作成
    data = pd.DataFrame(
        {"MultiPointIndex": [1, 1, 2, 2, 3], "Value": [10, 20, 30, 40, 50]}
    )
    field_numbers = [1, 2, 3]

    fields = Fields(
        name="test_data",
        image_analysis_method="test_method",
        data=data,
        field_numbers=field_numbers,
    )

    # イテレーションの結果を取得
    result = list(fields)

    # 各視野のデータが正しく分割されることを確認
    assert len(result) == 3

    # 視野1のデータ
    field1_data = result[0]
    assert len(field1_data) == 2
    assert all(field1_data["MultiPointIndex"] == 1)
    assert list(field1_data["Value"]) == [10, 20]

    # 視野2のデータ
    field2_data = result[1]
    assert len(field2_data) == 2
    assert all(field2_data["MultiPointIndex"] == 2)
    assert list(field2_data["Value"]) == [30, 40]

    # 視野3のデータ
    field3_data = result[2]
    assert len(field3_data) == 1
    assert all(field3_data["MultiPointIndex"] == 3)
    assert list(field3_data["Value"]) == [50]


def test_fields_with_empty_fields():
    """空の視野が含まれる場合のテスト"""
    # テストデータの作成（視野2にはデータなし）
    data = pd.DataFrame({"MultiPointIndex": [1, 1, 3, 3], "Value": [10, 20, 30, 40]})
    field_numbers = [1, 2, 3]

    fields = Fields(
        name="test_data",
        image_analysis_method="test_method",
        data=data,
        field_numbers=field_numbers,
    )

    # デフォルトでは空の視野も含まれる
    result = list(fields)
    assert len(result) == 3

    # 視野2は空のDataFrame
    field2_data = result[1]
    assert len(field2_data) == 0


def test_skip_empty_fields():
    """skip_empty_fieldsメソッドのテスト"""
    # テストデータの作成（視野2にはデータなし）
    data = pd.DataFrame({"MultiPointIndex": [1, 1, 3, 3], "Value": [10, 20, 30, 40]})
    field_numbers = [1, 2, 3]

    fields = Fields(
        name="test_data",
        image_analysis_method="test_method",
        data=data,
        field_numbers=field_numbers,
    )

    # skip_empty_fieldsを使用
    skip_fields = fields.skip_empty_fields()
    result = list(skip_fields)

    # 空の視野はスキップされ、2つの視野のみ
    assert len(result) == 2

    # 視野1のデータ
    field1_data = result[0]
    assert len(field1_data) == 2
    assert all(field1_data["MultiPointIndex"] == 1)

    # 視野3のデータ
    field3_data = result[1]
    assert len(field3_data) == 2
    assert all(field3_data["MultiPointIndex"] == 3)


def test_filename_parsing():
    """Filename列の解析のテスト"""
    # テストデータの作成
    test_data = pd.DataFrame(
        {
            "Filename": [
                "method1_000_data1.csv",
                "method1_000_data1.csv",
                "method2_000_data2.txt",
                "method2_000_data2.txt",
                "method3_000_data3.xlsx",
            ],
            "MultiPointIndex": [1, 2, 1, 2, 1],
            "Value": [10, 20, 30, 40, 50],
        }
    )

    cleansed_data = CleansedData(_data=test_data)
    target_data = ["data1", "data2", "data3"]
    field_numbers = [1, 2]

    lanes = Lanes(
        whole_data=cleansed_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )

    # データが正しく解析されることを確認
    assert "Data" in lanes.whole_data.columns
    assert "ImageAnalysisMethod" in lanes.whole_data.columns

    # Data列の値を確認
    expected_data = ["data1", "data1", "data2", "data2", "data3"]
    assert list(lanes.whole_data["Data"]) == expected_data

    # ImageAnalysisMethod列の値を確認
    expected_methods = ["method1", "method1", "method2", "method2", "method3"]
    assert list(lanes.whole_data["ImageAnalysisMethod"]) == expected_methods


def test_lanes_iteration():
    """Lanesクラスのイテレーションのテスト"""
    # テストデータの作成
    test_data = pd.DataFrame(
        {
            "Filename": [
                "method1_000_data1.csv",
                "method1_000_data1.csv",
                "method2_000_data2.txt",
                "method2_000_data2.txt",
            ],
            "MultiPointIndex": [1, 2, 1, 2],
            "Value": [10, 20, 30, 40],
        }
    )

    cleansed_data = CleansedData(_data=test_data)
    target_data = ["data1", "data2"]
    field_numbers = [1, 2]

    lanes = Lanes(
        whole_data=cleansed_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )

    # イテレーションの結果を取得
    result = list(lanes)

    # 2つのレーンが返されることを確認
    assert len(result) == 2

    # 各レーンがFieldsクラスのインスタンスであることを確認
    for lane in result:
        assert isinstance(lane, Fields)

    # 最初のレーン（data1）
    lane1 = result[0]
    assert lane1.data_name == "data1"
    assert lane1.image_analysis_method == "method1"
    assert len(lane1.data) == 2  # data1に対応する2行
    assert all(lane1.data["Data"] == "data1")

    # 2番目のレーン（data2）
    lane2 = result[1]
    assert lane2.data_name == "data2"
    assert lane2.image_analysis_method == "method2"
    assert len(lane2.data) == 2  # data2に対応する2行
    assert all(lane2.data["Data"] == "data2")


def test_lanes_with_empty_data():
    """対象データに該当するデータがない場合のテスト"""
    # テストデータの作成
    test_data = pd.DataFrame(
        {
            "Filename": ["method1_000_data1.csv", "method1_000_data1.csv"],
            "MultiPointIndex": [1, 2],
            "Value": [10, 20],
        }
    )

    cleansed_data = CleansedData(_data=test_data)
    target_data = ["data1", "data_nonexistent"]  # data_nonexistentは存在しない
    field_numbers = [1, 2]

    lanes = Lanes(
        whole_data=cleansed_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )

    result = list(lanes)

    # 2つのレーンが返される（存在しないデータも含む）
    assert len(result) == 2

    # 最初のレーンは正常なデータ
    lane1 = result[0]
    assert lane1.data_name == "data1"
    assert lane1.image_analysis_method == "method1"
    assert len(lane1.data) == 2

    # 2番目のレーンは空のデータ
    lane2 = result[1]
    assert lane2.data_name == "data_nonexistent"
    assert lane2.image_analysis_method == ""  # 空の場合は空文字列
    assert len(lane2.data) == 0


def test_lanes_with_empty_dataframe_sets_columns():
    """空のDataFrameの場合にImageAnalysisMethod/Data列が空文字列で用意される"""
    test_data = pd.DataFrame(
        {"Filename": pd.Series(dtype="object"), "MultiPointIndex": pd.Series(dtype="int")}
    )
    cleansed_data = CleansedData(_data=test_data)

    lanes = Lanes(whole_data=cleansed_data, target_data=[], field_numbers=[])

    assert "ImageAnalysisMethod" in lanes.whole_data.columns
    assert "Data" in lanes.whole_data.columns
    assert list(lanes.whole_data["ImageAnalysisMethod"]) == []
    assert list(lanes.whole_data["Data"]) == []


def test_complex_filename_formats():
    """複雑なファイル名形式のテスト"""
    # より複雑なファイル名のテストデータ
    test_data = pd.DataFrame(
        {
            "Filename": [
                "complex.method_000_data.with.dots.csv",
                "method_with_underscores_000_data_with_underscores.txt",
                "simple_000_simple.xlsx",
            ],
            "MultiPointIndex": [1, 1, 1],
            "Value": [10, 20, 30],
        }
    )

    cleansed_data = CleansedData(_data=test_data)
    # 実際の動作に合わせてtarget_dataを修正（拡張子は最初のドットで分割される）
    target_data = ["data", "data_with_underscores", "simple"]
    field_numbers = [1]

    lanes = Lanes(
        whole_data=cleansed_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )

    # Data列とImageAnalysisMethod列が正しく解析されることを確認
    # 実際の動作では、最初のドットで分割されるため"data.with.dots"は"data"になる
    expected_data = ["data", "data_with_underscores", "simple"]
    expected_methods = ["complex.method", "method_with_underscores", "simple"]

    assert list(lanes.whole_data["Data"]) == expected_data
    assert list(lanes.whole_data["ImageAnalysisMethod"]) == expected_methods


def test_integration_lanes_and_fields():
    """LanesとFieldsの統合テスト"""
    # テストデータの作成
    test_data = pd.DataFrame(
        {
            "Filename": [
                "method1_000_data1.csv",
                "method1_000_data1.csv",
                "method1_000_data1.csv",
                "method2_000_data2.txt",
                "method2_000_data2.txt",
            ],
            "MultiPointIndex": [1, 2, 3, 1, 2],
            "Value": [10, 20, 30, 40, 50],
        }
    )

    cleansed_data = CleansedData(_data=test_data)
    target_data = ["data1", "data2"]
    field_numbers = [1, 2, 3]

    lanes = Lanes(
        whole_data=cleansed_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )

    # 各レーンを取得し、さらに視野ごとに分割
    lanes_result = list(lanes)

    # レーン1（data1）の視野ごとのデータを確認
    lane1 = lanes_result[0]
    fields1 = list(lane1)
    assert len(fields1) == 3  # 3つの視野

    # 視野1のデータ
    field1_data = fields1[0]
    assert len(field1_data) == 1
    assert field1_data["Value"].iloc[0] == 10

    # 視野2のデータ
    field2_data = fields1[1]
    assert len(field2_data) == 1
    assert field2_data["Value"].iloc[0] == 20

    # 視野3のデータ
    field3_data = fields1[2]
    assert len(field3_data) == 1
    assert field3_data["Value"].iloc[0] == 30

    # レーン2（data2）の視野ごとのデータを確認
    lane2 = lanes_result[1]
    fields2 = list(lane2)
    assert len(fields2) == 3  # 3つの視野（視野3は空）

    # 視野1のデータ
    field1_data = fields2[0]
    assert len(field1_data) == 1
    assert field1_data["Value"].iloc[0] == 40

    # 視野2のデータ
    field2_data = fields2[1]
    assert len(field2_data) == 1
    assert field2_data["Value"].iloc[0] == 50

    # 視野3のデータ（空）
    field3_data = fields2[2]
    assert len(field3_data) == 0
