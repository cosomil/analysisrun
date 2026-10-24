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

DataFrameInputSnapshot は各入力 DataFrame の `Data` 列のみを使用してレーン分割する。
Pipeline の必須列は `Data` だけであり、`MultiPointIndex` / `ImageAnalysisMethod` /
`Filename` の存在・内容には依存しない。`Data` が欠けた場合は入力名を含む
ValueError とする。対象がゼロレーンの場合も必須列を検証する。
全入力 DataFrame について必須レーンの存在を初期化時に検証し、対象外レーンは除く。
スナップショット確定時と各復元時に `copy(deep=True)` し、NamedTuple 型を維持する。
object 列内の任意 Python オブジェクトまでの完全分離は保証しない。
`Filename` から `ImageAnalysisMethod` / `Data` を導出する責務は `read_context()` 側にあり、
既に正規化済みの列は尊重する。preprocess は正規化済み入力を受け取り、変換後も `Data`
列を保持する。Runner 自体は分割規則を知らない。入力正規化の scanner との共通化、
ExecutionContext の `Data` 必須チェックと公開 API への接続は #86〜#87 で行う。

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

## 子プロセス Map Executor（#84）

`analysisrun.pipeline_parallel` は内部の `MultiprocessMapExecutor` と
`file_snapshot` を提供する。Runner の注入点は変更しない。最低 Python は 3.13。
公開 Builder と `read_context()` への接続は #86〜#87 で行うため、現在は同じ
entrypoint で内部 Stage 列を再構築し、`execute_entrypoint()` へ渡す。
`pipeline-worker` モードでは、親の入力読込より先にこの入口へ分岐する。
入力・中間値のクラスは子でも import 可能なモジュールに定義する。

- `file_snapshot` は preprocess 後の入力をレーン ID 別の pickle ファイルに一度保存する。
  子には絶対パスだけを渡し、各 map と親の全体処理で毎回復元する。
  各 Segment の全 worker 回収後までファイルを保持し、正常・異常の両方で削除する。
- `MultiprocessMapExecutor(stages, entrypoint, on_image)` は全定義の Manifest を持つ。
  子は version、全 Manifest、半開区間、担当レーンを検証し、指定された map だけを実行する。
  連続 map のレーン内処理は逐次 Executor と共通で、中間 value は pickle 化しない。
  子では params と synthesized も各 map で独立して復元する。
- 並列上限は `os.process_cpu_count() or 1`。`ANALYSISRUN_MAX_WORKERS` は正の整数のみ許可。
  実プロセス数はレーン数との最小値。連続するレーン群を順序を保って均等分割し、
  Segment ごとに新規 worker を起動・回収する。ゼロレーンでは起動しない。
- 管理スレッドは stdin の要求 TAR を送信後、stdout を逐次受信し、別スレッドで stderr を回収する。
  子は要求を EOF まで読んでから応答を開始する。画像と value は有界 Queue（既定16）を通す。
  `on_image(data_name, image_name, bytes, image_type)` と Runner の `on_value` は
  呼出元のメインスレッドで同期実行し、到着時に反映する。figure の描画・エンコードは子で行う。
- barrier は全担当 value、正常 TAR 終端、全 worker の終了コード0を確認してから通過する。
  失敗時は未処理イベントの反映を止め、残る子を terminate、2秒で終了しなければ kill し、
  全通信スレッドを回収して例外を返す。受信済みの画像・value は取り消さない。
  `error.json`、重複・担当外・不足の value、壊れた pickle/TAR、異常終了を拒否する。

`execute_entrypoint` の `parallel-entrypoint` は画像を `output_dir` に保存し、
最終 DataFrame を返す。`parallel-entrypoint-streaming` は既存と同じ gzip TAR に
画像、`result_csv`、レーン別 `result_json` を書き、正常終端して `sys.exit(0)` する。
失敗時は最後のデータエントリに UTF-8 の `error` を書き、正常終端して非ゼロ終了する。
内部 worker の構造化 `error.json` とは別形式。Notebook は既定で逐次実行を選び、
従来の figure Output を渡す。pickle とバイナリ標準出力を要求しない。

小さい画像・value も即座に到着するよう、書込 TAR は write/tell/flush を持つ位置カウンタを
介して record バッファなしで送る。tarfile の非公開属性には触れない。
例外時には `TarFile.__exit__` が終端を書かないため、finally で明示的に close する。
画像名の衝突、巨大画像、エラー表示の詳細化は #85 で扱う。

`tests/test_pipeline_parallel.py` と同じ entrypoint を再実行する
`tests/pipeline_worker_entry.py` で並列性、結果等価性、barrier、到着時通知、状態分離、
Snapshot の共有・寿命、worker 停止、通信異常、外部 TAR の終了契約を検証する。
