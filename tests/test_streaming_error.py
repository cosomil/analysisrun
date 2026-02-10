"""
ストリーミングTAR出力のエラーハンドリングをテスト
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import BaseModel

from analysisrun.pipeable import (
    read_context,
    image_analysis_result_spec,
    entity_filter,
)
from analysisrun.tar import create_tar_from_dict, read_tar_as_dict
from analysisrun.scanner import Fields

DATA_DIR = Path(__file__).parent / "testdata"
IMAGE_ANALYSIS_RESULT_CSV = DATA_DIR / "image_analysis_result.csv"


class Params(BaseModel):
    threshold: int = 1


class ImageResults(NamedTuple):
    activity_spots: Fields = image_analysis_result_spec(
        description="Activity spots",
        cleansing=entity_filter("Activity Spots"),
    )


def _load_csv_df(path: Path) -> BytesIO:
    df = pd.read_csv(path)
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_streaming_error_outputs_partial_images_and_error_entry(monkeypatch):
    """
    ストリーミングTAR出力: エラー発生時に部分的な画像とerrorエントリが含まれることをテスト
    
    画像が生成された後にエラーが発生した場合、生成済みの画像とerrorエントリの両方が
    TAR出力に含まれることを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyze")
    stdout_buf = BytesIO()
    
    tar_buf = create_tar_from_dict(
        {
            "data_name": "0000",
            "sample_name": "SampleA",
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )
    
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )
    
    def analyze(args):
        # 最初の画像を出力
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        args.output(fig, "image1.png", "png")
        
        # 2番目の画像を出力
        fig2 = plt.figure()
        plt.plot([0, 2], [0, 2])
        args.output(fig2, "image2.png", "png")
        
        # エラーを発生させる
        raise ValueError("Test error during analysis")
    
    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)
    
    # エラー終了コードを確認
    assert excinfo.value.code == 1
    
    # TAR出力を確認
    stdout_buf.seek(0)
    tar_result = read_tar_as_dict(stdout_buf)
    
    # errorエントリが存在することを確認
    assert "error" in tar_result
    assert "解析処理中にエラーが発生しました" in tar_result["error"]
    
    # 部分的な画像が含まれることを確認
    assert "images" in tar_result
    images = tar_result["images"]
    assert "image1.png" in images
    assert "image2.png" in images
    
    # analysis_resultは含まれないことを確認（エラーで中断されたため）
    assert "analysis_result" not in tar_result


def test_streaming_tar_structure_order(monkeypatch):
    """
    ストリーミングTAR出力: 正常時のエントリ順序をテスト
    
    画像が先に出力され、analysis_resultが最後に出力されることを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyze")
    stdout_buf = BytesIO()
    
    tar_buf = create_tar_from_dict(
        {
            "data_name": "0000",
            "sample_name": "SampleA",
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )
    
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )
    
    def analyze(args):
        df = args.image_analysis_results.activity_spots.data
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        args.output(fig, "plot1.png", "png")
        
        fig2 = plt.figure()
        plt.plot([0, 2], [0, 2])
        args.output(fig2, "plot2.png", "png")
        
        return pd.Series({"total_value": int(df["Value"].sum())})
    
    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)
    
    assert excinfo.value.code == 0
    
    # TAR内容を確認
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    
    # 全てのエントリが存在することを確認
    assert "images" in tar_result
    assert "plot1.png" in tar_result["images"]
    assert "plot2.png" in tar_result["images"]
    assert "analysis_result" in tar_result
    
    # errorエントリは存在しないことを確認
    assert "error" not in tar_result
