# analysisrun
解析メソッドのベースとなるヘルパーを提供します。

## ライブラリ
### インストール方法
```shell
$ uv add git+https://github.com/cosomil/analysisrun --tag v0.0.6
```
※適切なバージョン（タグ）を選択してください。
※GitHubで公開されたパッケージをインストールするためには、システムにGitがインストールされている必要があります。

### 使用方法
```python
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

import analysisrun as ar


class Params(BaseModel):
    threshold: float = 0.8


class ImageAnalysisResults(NamedTuple):
    activity_spots: ar.Fields = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=ar.entity_filter("Activity Spots"),
    )


def analyze(
    args: ar.AnalyzeArgs[Params, ImageAnalysisResults],
) -> pd.Series:
    fields = args.image_analysis_results.activity_spots
    lane_name = fields.data_name

    # 各レーンの解析を実装する
    area_mean = fields.area.mean()
    ok = area_mean >= args.params.threshold

    fig, ax = plt.subplots()
    ax.hist(fields.area)
    ax.set_title(f"Area Histogram: {lane_name}")
    ax.set_xlabel("area")
    ax.set_ylabel("count")
    args.output(fig, f"{lane_name}_area_hist.png", "png")

    return pd.Series(
        {
            "ok": ok,
            "area_mean": area_mean,
        }
    )


def postprocess(args: ar.PostprocessArgs[Params]) -> pd.DataFrame:
    results = args.analysis_results.copy()
    if not results["ok"].all():
        results["ok"] = False
    return results


project_root = Path(".")
testdata_dir = project_root / "tests" / "testdata"

ctx = ar.read_context(
    Params,
    ImageAnalysisResults,
    manual_input=ar.ManualInput(
        Params(threshold=0.8),
        {
            "activity_spots": testdata_dir / "image_analysis_result.csv",
        },
        sample_names=testdata_dir / "samples.csv",
    ),
    output_dir=project_root / "output",
)

result = ctx.run_analysis(analyze=analyze, postprocess=postprocess)
result.to_csv(project_root / "output" / "result.csv", index=False)

```
