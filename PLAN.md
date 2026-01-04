# 実装計画 (SPEC.mdに基づく新runner/pipeable対応)

## ゴール
- 分散環境・並列ローカル・Notebook環境の3モードを統一的に扱う`AnalysisContext`/`read_context`を実装し、仕様どおりの入出力・エラーハンドリングを提供する。
- tarベースのIOでanalyze/postprocessを実行し、標準出力で結果を返す分散エントリポイントを実装する。
- ローカル実行時に手入力/自動スキャンの入力をバリデーションし、並列実行（tar経由のワーカー呼び出し）と画像出力ディレクトリ決定を行う。
- 既存API利用者に新APIを公開しつつ、旧`NotebookRunner`/`ParallelRunner`は非推奨として維持する。

## 方針・前提
- ImageAnalysisResultsはNamedTupleLikeで、各フィールドのデフォルトが`ImageAnalysisResultSpec`であることを必須チェックする。
- バリデーション/インタラクティブ入力はpydantic+既存`interactive`ユーティリティを使い、エラー時は`exit_with_error`でユーザ向けメッセージとスタックトレースを併せて出力する。
- 分散モードは`ANALYSISRUN_METHOD`で分岐し、標準出力にtarを書き終えたら必ず終了コード0で終了する。
- 並列ローカルは分散モードと同じtar契約でワーカー（サブプロセス）を起動し、Notebook環境はシーケンシャルに同じ処理を呼ぶ。

## 実装タスク
1. 型/ユーティリティ整備
   - `ImageAnalysisResultSpec`（descriptionとcleansingコールバック群の保持）と`entity_filter`ヘルパーを追加し、`__init__.py`で公開。
   - ImageAnalysisResultsの定義からpydanticの`create_model`で`ImageAnalysisResultsInput`を動的生成する仕組みを追加（デフォルト未設定時の例外を含む）。
   - tar入出力で使用するOutput実装（画像をバイト列化してtarに格納）と、ローカル用のファイル保存Outputを準備。
2. `read_context`の実装
   - `ANALYSISRUN_METHOD`・`get_interactivity()`・引数有無からモードを判定（analysis-only/postprocess-only/sequential/parallel-entrypoint）。
   - `manual_input`指定時は`InputModel`/`ManualInputModel`でバリデートし、未指定かつ対話可能なら`scan_model_input`で入力を促す。
   - VirtualFileを実データに読み込み、ImageAnalysisResultSpecのクレンジングを適用した`Lanes`を構築。画像出力先（manual/最初のファイルのディレクトリorカレント）決定を行う。
   - modeに応じた`AnalysisContext`内部状態（サンプル名対応表、stdin/stdoutハンドル、output_dir等）を保持。
3. 分散analyzeエントリポイント
   - `AnalysisInputModel`（ImageAnalysisResultsInputを型引数に挿入済み）で`read_tar_as_dict`結果をバリデートし、データフレーム化→クレンジング→Fields生成。
   - 画像出力をメモリ上に集約するOutputを渡して`analyze`を実行し、`analysis_result`+`images/*`を`create_tar_from_dict`でstdoutへ書き出して終了。
   - 例外やバリデーションエラー時は`exit_with_error`でメッセージとstacktraceを出力。
4. 分散postprocessエントリポイント
   - `PostprocessInputModel`でtar入力をバリデートし、解析結果DataFrameを読み込み。
   - `postprocess`（無指定時は素通し）実行後、`result_csv`と各レーンの`result_json/*`をtar化してstdoutへ出力。
5. ローカルオーケストレーション
   - サンプル名CSVを読み込み、レーン名→サンプル名のマッピングを作成。各レーンごとにanalyze用tarを生成。
   - Notebook（シーケンシャル）とターミナル（並列）で処理分岐。並列時はThreadPoolExecutorを使い、各スレッドでサブプロセスを起動してtar入出力を受け渡す（プロセス間pickle要件を避ける）。画像はoutput_dirへ保存。
   - 全レーンの分析結果をDataFrameへ結合し、postprocessを適用して戻り値とする。
6. エラーハンドリング/メッセージ
   - ユーザーフレンドリーな日本語メッセージを`exit_with_error`経由で出力しつつ、開発者向けにはスタックトレースをstderrへ出す。
   - 標準出力を結果専用にするため、解析処理中のprint等をstderrへリダイレクトする仕組みが必要なら追加。
7. テスト/確認
   - ImageAnalysisResultSpec/動的モデル生成/バリデーションの単体テスト。
   - analyze/postprocessのtar入出力の往復テスト（メモリ上で完結）。
   - モード判定・manual_input経路・output_dir決定ロジックのテスト。
   - 並列ワーカー呼び出しのスモークテスト（小さな疑似データでワークフローが通ることを確認）。

## 非スコープ・留意点
- 既存の`runner.NotebookRunner`/`ParallelRunner`は破壊的変更を避け、ドキュメントで非推奨扱いにとどめる。
- 外部オーケストレータやR2等のストレージ連携はモック前提で、当リポジトリ内で完結させる。
