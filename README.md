# analysisrun
解析メソッドのベースとなるヘルパーを提供します。

多段解析の公開 API は [型付き Pipeline API](docs/pipeline.md) を参照してください。

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
    args: ar.MapArgs[Params, ImageAnalysisResults, None, None],
) -> dict[str, object]:
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

    return {
            "data": args.data_name,
            "sample_name": args.sample_name,
            "analyzed_at": ar.get_jst_timestamp(),
            "ok": ok,
            "area_mean": area_mean,
        }


def postprocess(args: ar.SynthesizeArgs[Params, ImageAnalysisResults, dict[str, object], None]) -> pd.DataFrame:
    results = pd.DataFrame(args.values.values())
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

result = ctx.pipeline().map(analyze).finalize(postprocess)
result.to_csv(project_root / "output" / "result.csv", index=False)
```

### 多段 Pipeline のエラーと出力（内部実行 API）

`pipeline.execute_sequential` / `pipeline_entrypoint.execute_entrypoint` では、
最初の失敗で後続ステージを中断します。並列実行では Worker を停止・回収してから
例外を通知し、失敗時に最終 DataFrame は返しません。

- 親プロセス・Notebook の preprocess / map / synthesize / finalize の例外は元の型と
  traceback を維持します。例外の note にステージ種別と絶対インデックス（0 始まり）、
  map では lane_id・data_name・sample_name を付加します。
- Worker 内の失敗は内部 TAR の `error.json` で型・メッセージ・traceback・ステージ・
  レーンを伝達し、親で `TransferError` として通知します。転送・pickle 復元・非ゼロ終了の
  障害にも対象セグメント／レーンと子の stderr を含めます。
- 非 TAR の端末実行では stderr に traceback、続いて処理種別・対象と問い合わせ案内の
  cowsay を表示し、非ゼロ終了します。Notebook では元の例外を送出し、traceback の表示は Notebook に任せます。
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
  衝突回避は利用者の責務です。Notebook は指定された Output に委譲します。

画像は比較的小さく、1 レーン数枚程度を想定しています。BytesIO / bytes と有界 Queue で
転送するため、画像単位のメモリを使用します。チャンク転送や一時ファイルへの退避は行いません。

### 集計と前処理

`read_context() → ctx.pipeline() → finalize()` で解析を実行します。
map はレーン値（例: dict）を返し、最終 DataFrame の組み立ては
finalize で `pd.DataFrame(args.values.values())` のように行います。
`data` / `sample_name` 列が必要なら map の戻り値に明示的に含めます。

preprocess は定義の先頭で最大一度だけ設定でき、finalize の実行ごとに一度、
変換後の入力 NamedTuple を直接返します。共有情報は synthesize で返し、
後続の map / synthesize / finalize から `args.synthesized` で参照します。
[前処理の使用例](example/preprocess/README.md) を参照してください。

### 実行モード

- `notebook`: Notebook 内で逐次実行し DataFrame を返す。画像は Notebook 上に表示する。ManualInput が必須。
- `interactive`: TTY 端末で map を Worker により並列実行し DataFrame を返す。
  ManualInput があれば入力プロンプトは出さない。画像はファイルへ保存する。
- `streaming`: 非 TTY の stdin から TAR を読み、stdout へ gzip TAR を送って終了する。
  ManualInput があっても非 TTY では TAR 入力を要求する。
- `showschema`: `ANALYSISRUN_MODE=showschema` で入力 JSON Schema を出力し正常終了する。

`ANALYSISRUN_INTERACTIVITY=notebook|terminal` で環境判定を上書きできます。
内部 Worker は `ANALYSISRUN_PROCESS_ROLE=parallel-worker` で識別し、実行モードとは
分離します。Worker は元入力を読まず、親のスナップショットで指定 map 区間だけ実行します。
CSV・Params・サンプル名の定義と外部 TAR のリクエスト形式は維持します。
