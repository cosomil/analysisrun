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
    activity_spots: pd.DataFrame = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=[
            ar.entity_filter("Activity Spots"),
            ar.dropna("Area"),
        ],
    )


def analyze(
    args: ar.AnalyzeArgs[Params, ImageAnalysisResults],
) -> pd.Series:
    df = args.image_analysis_results.activity_spots
    # Fields は直接生成せず、scan_fields から復元する
    fields = ar.scan_fields(df, args.data_name)
    lane_name = fields.data_name

    # 各レーンの解析を実装する
    area_mean = df["Area"].mean()
    ok = area_mean >= args.params.threshold

    fig, ax = plt.subplots()
    ax.hist(df["Area"])
    ax.set_title(f"Area Histogram: {lane_name}")
    ax.set_xlabel("area")
    ax.set_ylabel("count")
    args.output(fig, f"{lane_name}_area_hist.png", "png")

    return pd.Series(
        {
            "analyzed_at": ar.get_jst_timestamp(),
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

### 多段 Pipeline のエラーと出力（内部実行 API）

`pipeline.execute_sequential` / `pipeline_parallel.execute_entrypoint` では、
最初の失敗で後続ステージを中断します。並列実行では Worker を停止・回収してから
例外を通知し、失敗時に最終 DataFrame は返しません。

- 親プロセス・Notebook の preprocess / map / synthesize / finalize の例外は元の型と
  traceback を維持します。例外の note にステージ種別と絶対インデックス（0 始まり）、
  map では lane_id・data_name・sample_name を付加します。
- Worker 内の失敗は内部 TAR の `error.json` で型・メッセージ・traceback・ステージ・
  レーンを伝達し、親で `TransferError` として通知します。転送・pickle 復元・非ゼロ終了の
  障害にも対象セグメント／レーンと子の stderr を含めます。
- 非 TAR の端末実行では stderr に traceback、続いて処理種別・対象と問い合わせ案内の
  cowsay を表示し、非ゼロ終了します。Notebook では元の例外を送出します。
- 受理済みの value / synthesized と参照先への変更はロールバックしません。
  ローカル保存済み・Notebook 表示済み・TAR 送信済みの画像も残ります。
- 外部 gzip TAR は `data_name/images/image_name` に画像を逐次送信します。
  画像・最終結果を含む全エントリ名の重複をエラーとします。同一レーンで同名画像を
  同一／別 map ステージから出力できません。別レーンの同名画像はパスが異なるため許可します。
  最終結果の `data` が重複する場合も `result_json` の名前重複で失敗します。
- 外部 TAR の失敗時は部分出力に続けて最後のデータエントリ `error`（UTF-8 メッセージ）を
  書き込み、TAR を正常終端して非ゼロ終了します。受信側は `error` の有無を必ず確認してください。
  出力先の切断など、物理的に書き込みを完結できない障害はこの保証の対象外です。
- ローカル保存先は従来どおり `output_dir / image_name` で、同名は上書きします。
  衝突回避は利用者の責務です。Notebook / sequential は指定された Output に委譲します。

画像は比較的小さく、1 レーン数枚程度を想定しています。BytesIO / bytes と有界 Queue で
転送するため、画像単位のメモリを使用します。チャンク転送や一時ファイルへの退避は行いません。
