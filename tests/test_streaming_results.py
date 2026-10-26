"""外部 streaming 結果の JSON 欠損値と既存 CSV 表現を検証する。"""

import json
import tarfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from analysisrun.pipeline_worker import _write_streaming_result_entries, writing_tar
from tests.support import read_tar_member


@pytest.mark.parametrize("include_data", [True, False])
def test_result_json_preserves_missing_values_and_stringifies_other_scalars(
    include_data: bool,
):
    frame = pd.DataFrame(
        {
            "mixed": pd.Series([None, np.nan, pd.NA, pd.NaT, "None", "nan"], dtype=object),
            "integer": pd.Series([1, pd.NA, 3, 4, 5, 6], dtype="Int64"),
            "float": pd.Series([1.5, 2.5, pd.NA, 4.5, 5.5, 6.5], dtype="Float64"),
            "boolean": pd.Series([True, False, pd.NA, True, False, True], dtype="boolean"),
            "string": pd.Series(["None", "nan", pd.NA, "", "日本語", "ok"], dtype="string"),
            "number": [1, 2, 3, 4, 5, 6],
            "flag": [True, False, True, False, True, False],
        }
    )
    if include_data:
        frame.insert(0, "data", [f"lane-{i}" for i in range(len(frame))])
    # 非連番・重複 index でも、欠損マスクを同じ行に適用する。
    frame.index = pd.Index([10, 10, 20, 20, 30, 30] if include_data else range(10, 16))
    original = frame.copy(deep=True)
    payload = BytesIO()
    with writing_tar(payload) as archive:
        _write_streaming_result_entries(archive, frame)

    expected_columns = {
        "mixed": [None, None, None, None, "None", "nan"],
        "integer": ["1", None, "3", "4", "5", "6"],
        "float": ["1.5", "2.5", None, "4.5", "5.5", "6.5"],
        "boolean": ["True", "False", None, "True", "False", "True"],
        "string": ["None", "nan", None, "", "日本語", "ok"],
        "number": ["1", "2", "3", "4", "5", "6"],
        "flag": ["True", "False", "True", "False", "True", "False"],
    }
    with tarfile.open(fileobj=BytesIO(payload.getvalue())) as archive:
        # CSV は astype(str) の既存動作（欠損値の文字列を含む）を維持する。
        expected_csv = original.astype(str).to_csv(index=False).encode()
        assert read_tar_member(archive, "result_csv") == expected_csv
        for position in range(len(frame)):
            name = f"lane-{position}" if include_data else str(frame.index[position])
            expected = {key: values[position] for key, values in expected_columns.items()}
            if include_data:
                expected["data"] = name
            assert json.loads(read_tar_member(archive, f"{name}/result_json")) == expected
    pd.testing.assert_frame_equal(frame, original)
