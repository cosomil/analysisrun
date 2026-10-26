# QC フィードバック

全体の集約結果を各レーンへ戻し、QC 判定を反映した最終画像を生成する実例です。
リポジトリのルートで実行します。

```bash
ANALYSISRUN_INTERACTIVITY=terminal uv run python -m example.qc-feedback
```

`input/measurements.csv` は画像解析結果（`MultiPointIndex,Entity,Value,Filename`）、
`input/samples.csv` はレーンとサンプル名（`data,sample`）です。
`QC_INPUT_DIR` / `QC_OUTPUT_DIR` で入力・出力ディレクトリを変更できます。

```python
result = (
    ctx.pipeline()
    .synthesize(calculate_reference)
    .map(analyze)
    .synthesize(evaluate_qc)
    .map(apply_qc_evaluation)
    .finalize(to_dataframe)
)
```

1. `calculate_reference` は QC と通常サンプルを区別せず、全レーンの測定値の中央値を基準値にします。基準値は正の値である必要があります。
2. `analyze` はレーンの平均測定値を基準値で割り、`minimum_score`（既定 0.8）以上なら `analysis_ok=True` とします。
3. `evaluate_qc` は `sample_name == "QC"` が 1 件以上あり、すべて解析合格なら検査全体を合格にします。QC 不在は不合格です。
4. `apply_qc_evaluation` は元の `analysis_ok` を保持し、`qc_ok` と `ok = analysis_ok and qc_ok` を確定してから、PASS/FAIL と QC 判定を記載した画像を出力します。
5. `to_dataframe` は全レーンの結果を DataFrame にします。

付属入力では中央値が 10、スコアが順に 1.0 / 1.0 / 1.2 / 0.4 となり、
QC 2 件が合格、通常サンプル A が合格、B が不合格になります。
QC の測定値を 1 件だけ 4 にすると、元の解析結果を保持したまま全レーンの最終判定が不合格になります。

対話端末では `output/result.csv` とレーン別の `output/0000_qc.png` 等を保存します。
Notebook では `importlib.import_module("example.qc-feedback.__main__").main()` で
DataFrame を取得でき、画像は Notebook に表示します。
非 TTY では入力 TAR を stdin に渡し、stdout の gzip TAR を保存します。

```bash
uv run python -m example.qc-feedback < request.tar.gz > response.tar.gz
```

入力 TAR は `params`（例: `{"minimum_score":0.8}`）、`sample_names`（CSV）、
`image_analysis_results/measurements`（CSV）を格納し、CSV には PAX ヘッダー
`is_file=true` を指定します。出力は `result_csv`、`{data}/result_json`、
`{data}/images/{data}_qc.png` です。JSON の欠損値は `null`、他の値は文字列になります。
失敗時は最終結果を返さず、送信済み画像に続いて最後の `error` エントリを出力し、非ゼロ終了します。
受信側は終了コードと `error` の有無を確認してください。
詳しい契約と実行できるクライアントは
[Streaming 入出力プロトコル](../../docs/streaming.md) を参照してください。

この例と同じエントリポイントを使用する [E2E](../../tests/test_qc_feedback.py) は、
QC 全件合格・一部不合格・不在、1 / 2 / 3 / 4 レーンを各モードで検証します。
画像出力直後の失敗はテスト用ラッパーで注入し、特定 Worker の処理順に依存せず部分画像の保持も確認します。
