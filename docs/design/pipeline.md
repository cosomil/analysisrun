# Pipeline の構築・実行モデル（#81）

対象: [#81](https://github.com/cosomil/analysisrun/issues/81)、親: [#78](https://github.com/cosomil/analysisrun/issues/78)。
状態所有権・転送・連続 map の Map Segment 化は [#82 の契約](pipeline-transfer.md) を参照。
同書が本書の単一 stage_index と連続 map 間のバリア案を更新する。

これは将来の公開 API の設計決定であり、以下の型・メソッドはまだ公開していない。
本変更で追加するコードは `tests/pipeline_probe.py` の独立実験のみ。

## 構築と実行の境界

- `read_context(Params, ImageResults)` の戻り値に `pipeline()` を追加する。
  新 API の実行コンテキスト名は `ExecutionContext[P, I]` とする。
  現 `AnalysisContext` からの改名・互換性は #86 で扱う。
- `pipeline()` は `InitialPipeline[P, I]` を返す。
  `preprocess()` を持つのはこの型のみ。最初の `preprocess` / `map` /
  `synthesize` の追加で `PipelineBuilder` を返す。
  両者は継承関係のない独立した公開型で、内部の immutable なステージ定義を共有する。
  `InitialPipeline` も `map` / `synthesize` / `finalize` を提供する。
- ビルダーは immutable なステージ定義の tuple を保持する。追加操作は新しい
  ビルダーを返し、元のビルダーを書き換えない。関数を呼び出さず、入力の変換、
  子プロセス起動、Output 出力も行わない。
- `finalize(func)` が終端定義を加え、検証後に実行を開始する。
  逐次モードでは `pd.DataFrame` を返す。既存のストリーミングモードと子モードは
  結果を送信してプロセスを終了するため、呼出元の後続コードには到達しない。
  `finalize` はビルダーを返さず、省略時には実行しない。
- 実行済みフラグ・履歴・Pipeline ID・レジストリは持たない。同じビルダーで
  再度 `finalize` を呼べば別の実行となり、前処理もその実行で一度行う。
  ただしマルチプロセスでは、一回のエントリポイント実行につき一つの Pipeline、
  一回の `finalize` に限定する。Notebook ではビルダーの再利用が可能。

## 型と前処理

`P` は params、`I` はレーンごとの元入力型、`V` はレーン値、`S` は全体合成結果。
入力 `I` と変換後入力 `J` は既存の `NamedTupleLike[pd.DataFrame]` に従う。
前処理は既存の `PreprocessArgs` と同様に、レーン分割前の cleansing 済み
全体データと対象レーンの対応表を一度受け取り、変換後の `J` を直接返す。
各 DataFrame 内のレーン分割は既存 scanner の規則に従う。

次は公開予定の型契約の抜粋（宣言の `...` は未実装）。
`PreprocessArgs`、`Output`、`NamedTupleLike` は既存型を使用する。

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

@dataclass
class MapArgs[P: BaseModel, I: NamedTupleLike[pd.DataFrame], V, S]:
    params: P
    data_name: str
    sample_name: str
    image_analysis_results: I
    value: V
    synthesized: S
    output: Output

@dataclass
class SynthesizeArgs[P: BaseModel, I: NamedTupleLike[pd.DataFrame], V, S]:
    params: P
    image_analysis_results: Mapping[str, I]
    values: Mapping[str, V]
    synthesized: S

class PipelineBuilder[P: BaseModel, I: NamedTupleLike[pd.DataFrame], V, S]:
    def map[N](
        self, func: Callable[[MapArgs[P, I, V, S]], N]
    ) -> "PipelineBuilder[P, I, N, S]": ...

    def synthesize[N](
        self, func: Callable[[SynthesizeArgs[P, I, V, S]], N]
    ) -> "PipelineBuilder[P, I, V, N]": ...

    def finalize(
        self, func: Callable[[SynthesizeArgs[P, I, V, S]], pd.DataFrame]
    ) -> pd.DataFrame: ...

class InitialPipeline[P: BaseModel, I: NamedTupleLike[pd.DataFrame]]:
    def preprocess[J: NamedTupleLike[pd.DataFrame]](
        self, func: Callable[[PreprocessArgs[P, I]], J]
    ) -> PipelineBuilder[P, J, None, None]: ...

    def map[V](
        self, func: Callable[[MapArgs[P, I, None, None]], V]
    ) -> PipelineBuilder[P, I, V, None]: ...

    def synthesize[S](
        self, func: Callable[[SynthesizeArgs[P, I, None, None]], S]
    ) -> PipelineBuilder[P, I, None, S]: ...

    def finalize(
        self, func: Callable[[SynthesizeArgs[P, I, None, None]], pd.DataFrame]
    ) -> pd.DataFrame: ...

class ExecutionContext[P: BaseModel, I: NamedTupleLike[pd.DataFrame]]:
    def pipeline(self) -> InitialPipeline[P, I]: ...
```

`preprocess` による `I → J` は元入力の型変換であり、`value` の初期化ではない。
後続の `image_analysis_results` は変換後の入力を指す。
map / synthesize は元入力を置換せず、読み取り専用として扱う。
全体の中間値は `synthesized` が担い、必要なら先頭の synthesize で生成する。
新 Pipeline の型・Args・状態・転送に `extra` / `ProcessedInputs` は導入しない。
既存の `run_analysis_with_preprocess` の互換性のため、それらの既存型は残す。
Mapping の外側を保護しても DataFrame 等の内部まで immutable にはならないため、
共有値の破壊的変更によってレーン間の結果を変える実装は契約外とする。

前処理に渡す `targets: Mapping[str, str]` のキー（data_name）、sample_name、順序を
実行中維持する。前処理結果の各 DataFrame は全対象レーンを scanner で分割できる
必要があり、欠損・改名は map 開始前にエラーとする。対象外の行から新たなレーンを
追加しない。各レーン内の行の削減・列追加・値変換は許容する。
前処理後は cleansing を再適用せず、変換済み入力を後続へ引き渡す。
具体的な空レーン検証・転送形式は #82 / #87 で整備する。

### 使用例

以下の `Params`、`ImageResults`、`PreparedResults` は利用者の入力型、
`Features`、`QC` は pickle 化可能な dataclass 等の中間型とする。

```python
def extract(
    args: MapArgs[Params, ImageResults, None, None],
) -> Features: ...

def qc(
    args: SynthesizeArgs[Params, ImageResults, Features, None],
) -> QC: ...

def render(
    args: SynthesizeArgs[Params, ImageResults, Features, QC],
) -> pd.DataFrame: ...

ctx = read_context(Params, ImageResults)
result = ctx.pipeline().map(extract).synthesize(qc).finalize(render)
```

```python
def prepare(
    args: PreprocessArgs[Params, ImageResults],
) -> PreparedResults: ...

def extract_prepared(
    args: MapArgs[Params, PreparedResults, None, None],
) -> Features: ...

def qc_prepared(
    args: SynthesizeArgs[Params, PreparedResults, Features, None],
) -> QC: ...

def correct(
    args: MapArgs[Params, PreparedResults, Features, QC],
) -> float: ...

def render_prepared(
    args: SynthesizeArgs[Params, PreparedResults, float, QC],
) -> pd.DataFrame: ...

result = (
    ctx.pipeline()
    .preprocess(prepare)
    .map(extract_prepared)
    .synthesize(qc_prepared)
    .map(correct)
    .finalize(render_prepared)
)
```

禁止例（静的には属性・引数の型エラー、動的には属性不在の `AttributeError`）:

```python
ctx.pipeline().preprocess(prepare).preprocess(prepare)
ctx.pipeline().map(extract).preprocess(prepare)
ctx.pipeline().synthesize(initial_summary).preprocess(prepare)
# この引数形式は採用しない（TypeError）。
ctx.pipeline(preprocess=prepare)
```

内部の定義検証でも先頭以外・複数の preprocess は `ValueError` にする。
元の `initial = ctx.pipeline()` から複数のビルダーを作ること自体は可能で、
「先頭で一度」は確定した各パイプライン定義につき最大一回の制約。
元の InitialPipeline の消費状態は管理しない。
ジェネリック型の実検証と公開実装は #86 の対象。

## 状態遷移と実行モード

実行状態は `image_analysis_results` / レーン別 `values` / `synthesized` の3要素。
params・対象レーンの識別子は別途保持する。

| 操作 | 更新するもの | 保持するもの |
| --- | --- | --- |
| 初期化 | 全対象レーンの value = None、synthesized = None | 対象集合・識別子 |
| preprocess（省略可） | 元入力 | value / synthesized は None のまま |
| map | 各レーンの value を戻り値で置換 | 元入力・synthesized |
| synthesize | synthesized を戻り値で置換 | 元入力・全レーンの values |
| finalize | 全体引数から最終 DataFrame を生成 | 後続ステージなし |

map / synthesize はそれぞれゼロ回・連続・どちらからの開始も許容する。
`ctx.pipeline().finalize(func)` も有効で、func は全レーンの None 値を受け取る。
map / synthesize の戻り値としての `None` も合法で、そのまま値を置換する。
対象レーンがゼロなら map は呼ばず、全体処理は空の Mapping を受け取る。
finalize は必須 callable で、None 等の戻り値を暗黙に前の結果へ置き換えない。
戻り値の DataFrame 検証は #83 で実装する。

マルチプロセスでは preprocess / synthesize / 最終整形を親で行い、各 map の
全対象レーンが完了してから次へ進む。連続 map にもステージ境界がある。
preprocess は先頭で一回のみ実行し、子では定義の再構築のみを行う。
map は他レーンを参照しない処理契約であり、マルチプロセスエンジンへ並列処理を
委ねる。実際の同時実行数はレーン数・資源・設定に依存し、2以上を保証しない。
Notebook は同じ意味論を逐次実行する。#78 の「必ず並列」はこの区別へ改める。

## エントリポイント再実行の識別

ステージは preprocess を含む先頭から **0 始まりの絶対インデックス**で数える。
finalize も Manifest に含める。例:

| preprocess | ステージ列 | 二つ目の map |
| --- | --- | --- |
| あり | preprocess, map, synthesize, map, finalize | 3 |
| なし | map, synthesize, map, finalize | 2 |

Manifest は version と順序付きの `(kind, callable.__module__, callable.__qualname__)`
を JSON のソート済みキー・空白なし区切りで canonical 化する。
比較は全文一致とし、ハッシュの選定は不要。子は同じエントリポイントを再実行し、
`finalize` に到達してから次の順に処理する。

1. 再構築した全定義の Manifest を比較する。
2. 指定インデックスが整数・範囲内・map であることを検証する。
3. 一致した callable のみ、受け取った計算済みレーン状態に適用して結果を返す。

子は前段の map、preprocess、synthesize、最終整形を呼ばない。
制御メッセージの最小項目は `manifest`（その version を含む）、`stage_index`、
対象の `data_name` / `sample_name` とステージ開始時の必要状態（params、
変換済み元入力、value、synthesized）。対象は将来チャンク化可能。
実験では対象レーン一つのダミー数値を `state` にまとめ、JSON を TAR/stdin で送る。
これは本番 pickle プロトコルの先行実装ではない。
子の照合失敗は非ゼロ終了と読めるエラーで親に通知し、誤った関数を呼ばない。

### 実験の実行と観測

実験では公開ビルダー型を実装せず、内部定義に相当する `Definition` のみを使う。
元入力は DataFrame の代わりに整数で表し、前処理はその変換結果を直接返す。

```bash
uv run pytest tests/test_pipeline_probe.py -v
```

検証結果: `uv run ruff check` は `All checks passed!`、`uv run pytest` は
100 passed（うち本実験のテスト22件）。追加した Python ファイルの
`uv run ruff format --check` も通過した。

`run_child` は `subprocess.Popen([sys.executable, entrypoint], ...)` と既存の
`create_tar_from_dict` / `read_tar_as_dict` を使う。各テストは終了コード・stderr・
応答内の実呼出し記録を検証し、正常時は親子の Manifest と指定インデックスも比較する。

- 前処理あり・なしのいずれも、二つ目の map 指定で呼ばれた関数は `apply_qc` のみ。
  変換済み元入力=4、計算済み value=8、synthesized=12 を渡して value=24 を得る。
  逐次実行でも、元入力=3 を前処理で4に変換すると、後続の map / synthesize が
  その入力を参照して value=24 / synthesized=12 を得る（前処理なしは18 / 9）。
- 追加・削除・順序・種別・同種 callable・finalize callable・前処理の有無を
  子だけ変えると、いずれも `pipeline manifest mismatch`、終了コード1、呼出しなし。
- 不正なインデックスおよび preprocess / synthesize / finalize 指定でも呼出しなし。
- 構築のみでは関数を呼ばず、元の定義を変更しない。逐次実行は Manifest 生成さえ
  不要で、前処理を一回だけ呼ぶ。ゼロ段、synthesize 開始、同種の連続も確認する。

### 識別方法の限界

直接起動の `__main__` は固定の `<entrypoint>` へ正規化する。実験をテストから
import した場合の `tests.pipeline_probe` も同じ名前にするが、本番では
実行コンテキストが把握するエントリポイントモジュールだけを正規化する。
他のモジュールを無差別に正規化してはならない。

module / qualname は関数本体や値の fingerprint ではない。同じローカル関数の
異なるクロージャ、lambda、ラッパー、同じ qualname を設定した callable は
衝突し得る。グローバル変数、環境、関数本体の変更も検出できない。
マルチプロセスの初期実装は、安定した module / qualname を持つ名前付きの
トップレベル関数をサポート範囲とし、匿名関数・クロージャ・partial・callable
インスタンスは対象外にする。逐次モードはこの識別制限を必要としない。
同じ定義順序を親子で再構築し、入力や実行モードで定義を分岐しないことを要求する。
エントリポイントのトップレベルコード自体の副作用をこの仕組みで抑止はできない。

明示 ID はラッパー等を扱いやすい一方、ID の使い回しを検出できない。
source fingerprint はソース取得不能・デコレータ・依存グローバルを扱い切れない。
現段階は構造と識別子の照合を採用し、計算内容の同一性保証を追加しない。
この実験は一レーンの選択実行の成立性を示すもので、実並列度・DataFrame の型検証・
pickle 復元・バリア・Output・エラー集約の本実装の検証は含まない。

## #78 および後続 Issue への引き継ぎ

#78 の実装計画に反映する設計判断は次のとおり（GitHub 更新用の要約）。

- #81: immutable な定義構築、`InitialPipeline` のみの preprocess、finalize での
  実行開始を採用。両公開型は継承せず内部定義を共有する。
  全体前処理は変換後入力を直接返し、親で一回実行する。
  独立プロトタイプで親子 Manifest の一致・二つ目の map のみの実行・不一致の
  実行前拒否を確認した。型付き使用例・制約・限界は本設計メモを参照。
- #78 の実行モデル: 「map は必ず並列」を、レーン独立の契約とエンジンの並列実行に
  分ける。Notebook は逐次、実際の並列度の下限保証は設けない。
- #82: レーン集合・識別子と型の推移を固定し、最小制御情報を Manifest / 絶対
  インデックス / 対象 / ステージ開始時状態とする。変換済み入力・values・
  synthesized の pickle 転送とバージョンを設計する。
- #83: ゼロ段・同種連続・synthesize 開始・None 初期値を含む逐次意味論を実装する。
- #84: 同一エントリポイントの finalize で要求を解釈し、全定義照合後に指定 map のみを
  実行する。一つのエントリポイントにつき一つの Pipeline とし、ID は導入しない。
  全 map 境界のバリアと親子のエラー伝搬はこの実験とは別に実装する。
- #85: Output、失敗集約、ストリーミング・子モードの終了契約を実装する。
- #86: 本メモの InitialPipeline / PipelineBuilder と Args のジェネリック検証、
  不正な preprocess の静的・実行時拒否を完成させる。両公開型の独立性、
  PipelineBuilder / Args の4要素型、ExecutionContext への改名・互換性を扱う。
- #87: 既存 API の互換性を維持しつつ、新 Pipeline の前処理は変換後入力を直接返す
  形式へ移行する。旧 API の extra を新状態へ暗黙に移さず、全体の中間値が必要なら
  synthesize へ移す案を示す。対象欠損の検証も整備する。
  既存の run_analysis 系 API は本変更では変更しない。
