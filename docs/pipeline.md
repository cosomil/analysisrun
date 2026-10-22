# 型付き Pipeline API

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
既存の画像保存先・TAR 出力に接続する。

## 型と前処理

公開型は `PipelineBuilder[P, I, V, S]`、`MapArgs[P, I, V, S]`、
`SynthesizeArgs[P, I, V, S]`。`P` は BaseModel の Params、`I` は DataFrame の
NamedTuple、`V` は各レーン値、`S` は全体の合成値。
`map()` の戻り値は V のみ、`synthesize()` の戻り値は S のみを更新する。
初期 V / S はともに None。両ステージは連続・交互・ゼロ回で組み合わせられ、
None を返して状態を置換することもできる。

`InitialPipeline[P, I]` のみが `preprocess(func)` を提供する。
`func: Callable[[PreprocessArgs[P, I]], J]` はレーン分割前の全体入力と targets を
受け取り、変換後の DataFrame NamedTuple を直接返す。後続の入力型は J に変わる。
元入力は Context に残る。新 API の共有追加情報は synthesize の戻り値として扱い、
旧 API の `ProcessedInputs` / `extra` を使用しない。

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

旧 `run_analysis` 系メソッドと専用型は #87 で削除した。互換ラッパーは設けない。
入力形式・実行モード・移行手順は [README](../README.md#実行モードと旧-api-からの移行) を参照。
クラス名は ExecutionContext に変更し、AnalysisContext の互換エイリアスは設けない。
