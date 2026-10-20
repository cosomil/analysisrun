# Pipeline 共通 Runner と逐次実行（#83）

`src/analysisrun/pipeline.py` は、[#81](pipeline.md) のステージ列と
[#82](pipeline-transfer.md) の状態契約を実行する内部エンジン。
公開ビルダー、ExecutionContext への接続、既存 API の移行は後続 Issue で行う。
既存の `run_analysis` / `run_analysis_with_preprocess` は変更しない。

## 呼び出しと責務

```python
from analysisrun.pipeline import Stage, execute_sequential

result = execute_sequential(
    [
        Stage("preprocess", prepare),  # 省略可。先頭に一度だけ
        Stage("map", extract),
        Stage("synthesize", qc),
        Stage("map", correct),
        Stage("finalize", to_dataframe),
    ],
    params=params,
    image_analysis_results=inputs,
    targets={"data_b": "sample B", "data_a": "sample A"},
    output=output,
)
```

Runner は定義を検証して対象順に連続した内部レーン ID を割り当てる。
preprocess はコピーした全体入力、deepcopy した params、外側 read-only の targets
を既存 PreprocessArgs で受け取る。戻り値の NamedTupleLike を後続入力とする。
map / synthesize / finalize 用の Args は設計のフィールドを持つ内部型であり、
公開ジェネリック型の実装は #86 に委ねる。

`execute_pipeline` には MapExecutor と SnapshotFactory を個別に渡す。
`execute_sequential` は SequentialMapExecutor と dataframe_snapshot を組み合わせる。
SnapshotFactory は `(inputs, lanes)` を受け取る context manager の factory。
preprocess 完了直後に一度生成し、成功・例外のどちらでも実行終了後に解放する。
Executor は子の停止・回収も含めて完了してから return / raise する必要がある。
この境界により #84 のファイル Snapshot と並列 Executor に差し替えられる。
共通 InputSnapshot Protocol は `restore_all()` だけを要求する。
逐次 Executor 向けの追加 Protocol が `restore(lane_id)` を要求する。

DataFrameInputSnapshot は既存 scanner の分割規則を利用する。
全入力 DataFrame について必須レーンの存在を初期化時に検証し、対象外レーンは除く。
スナップショット確定時と各復元時に `copy(deep=True)` し、NamedTuple 型を維持する。
object 列内の任意 Python オブジェクトまでの完全分離は保証しない。
scanner / ExecutionContext の製品統合は #87 の範囲。

## 更新と失敗

Runner はステージ列をその場で走査し、連続 map を半開区間 `[start, end)` の
MapSegment にまとめる。インデックスは preprocess / finalize を含む全定義の絶対位置。
一実行で同じ Executor インスタンスを使う。
逐次 Executor は一レーンの Segment 内 map をすべて実行してから次レーンへ進む。
各 map へ復元した入力と deepcopy した params を渡し、中間 value は直接引き継ぐ。
pickle は使用しない。synthesized も参照渡しとなる。

レーンの最終 map が成功したら同期 `on_value(lane_id, value)` で即座に更新する。
None も有効な更新。未完了レーンの途中戻り値は Runner に反映しない。
ただし元 value オブジェクト自身への破壊的変更は取り消さない。
例外はそのまま伝播し、後続ステージを実行しない。完了済みレーン値、Output の
副作用、参照共有したオブジェクトの変更はロールバックしない。
Output は各 MapArgs へ渡す別経路であり、value の通知にバッファしない。

synthesize / finalize は復元した全入力と values の外側 read-only Mapping、
deepcopy した params、参照の synthesized を受け取る。
要素と synthesized の変更禁止は契約であり、実行時には強制しない。
synthesize の戻り値で synthesized を置換し、finalize は DataFrame を返す必要がある。
ゼロレーンでも全体処理は空 Mapping を受け取って実行する。

`tests/test_pipeline.py` は QC feedback、入力・params の分離、参照の保持、初期値、
None 更新、ゼロレーン、失敗時の結果保持と中断、同期 callback、スナップショット寿命、
Executor と Snapshot の独立した注入を検証する。
