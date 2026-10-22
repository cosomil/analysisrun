# preprocess example

`read_context() → pipeline().preprocess().synthesize().map().finalize()` の移行例です。
対話端末以外から手動入力で実行する場合は環境判定を指定します。

```bash
ANALYSISRUN_INTERACTIVITY=terminal uv run python -m example.preprocess
```

- preprocess は finalize 時に親で一度実行し、`scaled_value` / `scaled_ratio` を追加した入力 NamedTuple を直接返す。
- synthesize は前処理済みの全対象レーンから共有の `target_count` を生成する。
- map は前処理済み入力と `args.synthesized` を参照し、レーンの集計を dict で返す。
- scan_fields は正規化済み DataFrame から Fields を復元する。
- finalize は `args.values` から DataFrame を組み立て、threshold と is_valid を追加する。

旧 `ProcessedInputs` / `extra` は削除しました。入力変換は preprocess、共有値は
synthesize に移してください。preprocess 後に cleansing は再適用されません。
結果は `example/preprocess/output/result.csv` に出力されます。
