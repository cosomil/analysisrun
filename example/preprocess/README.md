# preprocess example

`preprocess` を使って `run_analysis_with_preprocess` を実行する最小例です。

```bash
uv run python -m example.preprocess
```

この例では次の流れを確認できます。

- `preprocess`: CSV から読み込んだ DataFrame に `scaled_value` と `scaled_ratio` を追加する
- `preprocess`: `extra` として対象レーン数（`target_count`）を返す
- `analyze`: 前処理済み DataFrame と `extra` を使ってレーン単位の集計を行う
- `scan_fields`: 必要なときだけ正規化済み DataFrame から一貫した検証付きで `Fields` を復元する
- `postprocess`: 全レーンの解析結果に `threshold` を付与する

結果は `example/preprocess/output/result.csv` に出力されます。
