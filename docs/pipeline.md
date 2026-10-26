# 型付き Pipeline API

実用例は [QC フィードバック](../example/qc-feedback/README.md) を参照。
全レーンの基準値 → レーン解析 → QC 判定 → 最終画像・結果という構成を、
Notebook・interactive・streaming で同じ定義のまま実行できる。

`read_context(Params, ImageResults)` は `ExecutionContext[Params, ImageResults]` を返す。
`ctx.pipeline()` から定義を組み立て、`finalize()` で実行する。
`ExecutionContext`、`InitialPipeline`、`PipelineBuilder`、`MapArgs`、
`SynthesizeArgs`、`PreprocessArgs`、`Output` は `analysisrun` から import できる。
パッケージには `py.typed` を含める。

```python
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel

from analysisrun import (
    MapArgs, SynthesizeArgs, entity_filter, image_analysis_result_spec, read_context,
)


class Params(BaseModel):
    threshold: float = 1.0


class ImageResults(NamedTuple):
    cells: pd.DataFrame = image_analysis_result_spec(
        description="Cells", cleansing=entity_filter("Activity Spots"),
    )


def extract_features(args: MapArgs[Params, ImageResults, None, None]) -> float:
    return float(args.image_analysis_results.cells["Value"].mean())


def evaluate_qc(args: SynthesizeArgs[Params, ImageResults, float, None]) -> float:
    return max(args.values.values(), default=args.params.threshold)


def apply_qc(args: MapArgs[Params, ImageResults, float, float]) -> bool:
    return args.value >= args.synthesized


def summarize_results(args: SynthesizeArgs[Params, ImageResults, bool, float]) -> int:
    return sum(args.values.values())


def to_dataframe(args: SynthesizeArgs[Params, ImageResults, bool, int]) -> pd.DataFrame:
    return pd.DataFrame({"data": list(args.values), "passed": list(args.values.values())})


ctx = read_context(Params, ImageResults)
result = (
    ctx.pipeline()
    .map(extract_features)
    .synthesize(evaluate_qc)
    .map(apply_qc)
    .synthesize(summarize_results)
    .finalize(to_dataframe)
)
```

Notebook では既存の `ManualInput` を `read_context()` に渡す。
通常スクリプトでは親が入力を読み込み、map の区間を子プロセスで実行する。
外部 TAR モードでは最終結果を送信して終了する。Notebook と通常スクリプトの
`finalize()` は DataFrame を返す。`args.output(fig, name, image_type, **kwargs)` は
Notebook では画面表示、対話端末ではファイル保存、外部 TAR モードでは TAR 出力を行う。
Notebook の例外は元のまま伝播し、traceback の表示は Notebook に任せる。

## 型と前処理

公開型は `PipelineBuilder[P, I, V, S]`、`MapArgs[P, I, V, S]`、
`SynthesizeArgs[P, I, V, S]`。`P` は BaseModel の Params、`I` は DataFrame の
NamedTuple、`V` は各レーン値、`S` は全体の合成値。
`map()` の戻り値は V のみ、`synthesize()` の戻り値は S のみを更新する。
初期 V / S はともに None。両ステージは連続・交互・ゼロ回で組み合わせられ、
None を返して状態を置換することもできる。

`map` は各レーンの入力、直前のレーン値 `args.value`、直前の集約値
`args.synthesized` を受け取る。`synthesize` は全レーンの入力
`args.image_analysis_results` と値 `args.values`、直前の集約値を受け取り、親で実行する。
任意の dataclass 等を返せるが、並列転送する値は pickle 化可能である必要がある。
`finalize` も全レーンの値と直前の集約値を受け取り、必ず DataFrame を返す。
結果の列や `data` / `sample_name` は利用者が明示的に組み立てる。

全体の QC 判定を画像へ反映する場合は、判定の `synthesize` の後に
画像生成の `map` を置く。既に出力した画像は後から更新されない。
QC を `all()` で判定する際は、QC 不在時の扱いも明示する。

`InitialPipeline[P, I]` のみが `preprocess(func)` を提供する。
`func: Callable[[PreprocessArgs[P, I]], J]` はレーン分割前の全体入力と targets を
受け取り、変換後の DataFrame NamedTuple を直接返す。後続の入力型は J に変わる。
元入力は Context に残る。共有情報は synthesize の戻り値として扱う。
preprocess は finalize の実行ごとに親で一度だけ実行され、後続の map 区間や
Worker の起動時には再実行されない。

前処理後の各 DataFrame は全対象レーンの `Data` 列を含む必要がある。
`read_context()` は既存 CSV の Filename からの正規化を維持するが、前処理後には
scanner や cleansing を再適用しない。後続のレーン分割は Data 列のみで行う。

二重 preprocess や map / synthesize 後の preprocess は、静的には属性エラー、
実行時には AttributeError になる。内部 Runner も不正なステージ列を検証する。
ビルダーは immutable であり、構築時に利用者の関数は呼ばない。
初期ビルダーからの分岐や Notebook での再実行は独立した実行になる。
マルチプロセスのエントリポイントでは一つの Pipeline を一度 finalize する。

型注釈のない関数も実行できるが、静的な型保証は弱くなる。
実行時に型引数を復元・検証せず、pickle は信頼できるプロセス間の転送にのみ使用する。
転送境界の値は pickle 化可能であることが必要。子は同じ定義を再構築し、Manifest
照合後に指定 map 区間のみを実行する。子の Context は入力を読み込まないため、
定義構築を `ctx.params` や実行モードで分岐させず、データはステージ引数から参照する。

型推論と前処理の型変更の完全な例は
[成功例](../tests/typing/pipeline_valid.py)、拒否する例は
[失敗例](../tests/typing/pipeline_invalid.py) を参照。
`uv run pytest tests/test_pipeline_typing.py` は basedpyright で成功例を検証し、
失敗例の診断位置・種類を確認する。

集計・前処理の使い方と実行モードは [README](../README.md#集計と前処理) を参照。

## 実行・失敗時の契約

| モード | map の実行 | finalize の結果 | 画像 |
| --- | --- | --- | --- |
| notebook | 同一プロセスで逐次 | DataFrame を返す | Notebook に表示 |
| interactive | Worker で並列 | DataFrame を返す | output_dir に保存 |
| streaming | Worker で並列 | gzip TAR を stdout に送り終了 | TAR に逐次格納 |

最初の失敗で後続ステージを中断する。Notebook は元の例外と traceback を維持する。
interactive は stderr に traceback → cowsay の順で診断を出し、非ゼロ終了する。
Worker の失敗は親で `TransferError` として通知されるため、元例外の型は直接伝播しない。
streaming の失敗は送信済み部分画像を保持し、最後に `error` を格納して gzip TAR を
正常終端し、非ゼロ終了する。送信先の切断など、書き込み不能な障害は終端保証の対象外。
表示・保存済みの画像や受理済みの値への変更はロールバックしない。

streaming の最終出力は `result_csv` と `{data}/result_json`（欠損値は `null`、他の値は文字列）、
画像は `{data}/images/{image_name}`。最終結果を含む TAR エントリ名の重複はエラー。
ローカル画像は `output_dir / image_name` に保存され、同名は上書きするため、
ステージやレーンごとに名前を分ける。

入力 TAR・schema・出力形式・終了コードとクライアントの実行例は
[Streaming 入出力プロトコル](streaming.md) を参照。

旧 `run_analysis` API は未リリースのため、互換ラッパーは提供しない。
実行入口は `read_context()`、レーン処理は `map()`、全体集約は `synthesize()`、
最終 DataFrame の生成は `finalize()` で定義する。
内部設計は [設計資料](design/pipeline.md)、[転送](design/pipeline-transfer.md)、
[Runner](design/pipeline-runner.md) を参照。
