# ベンチマーク

リポジトリのルートで実行します。

```bash
uv run example/benchmark
```

逐次実行（Notebook モード）と通常端末の並列実行（interactive）を各1回測定し、
時間と高速化率を表示します。各実行後に CSV と画像を `golden/` と比較します。
逐次実行では比較用の画像保存処理を指定し、Notebook の画面表示を使いません。
出力は一時ディレクトリに保存し、測定終了後に削除します。

`ANALYSISRUN_MAX_WORKERS=2 uv run example/benchmark` で Worker 数の上限を指定できます。
既定の入力は2レーンのため、使用する Worker 数は最大2です。
このスクリプトは streaming の測定には対応していません。
