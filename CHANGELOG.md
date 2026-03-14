# Changelog

## [v0.0.7](https://github.com/cosomil/analysisrun/compare/v0.0.6...v0.0.7) - 2026-03-14
- feat(analysisrun): Entity列を複数の値でフィルタリングできるようにする by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/41
- feat(analysisrun): interactiveモジュールのブラッシュアップ  by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/43
- feat(analysisrun): スクリプトの実行環境の情報を取得する内部用ユーティリティを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/44
- feat(analysisrun): NamedTupleLikeを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/45
- feat(analysisrun): マルチバイト文字対応のcowsayを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/46
- Redirect stdout to stderr during analysis/postprocess execution by @Copilot in https://github.com/cosomil/analysisrun/pull/48
- Add double quote trimming to VirtualFile, FilePath, and DirectoryPath by @Copilot in https://github.com/cosomil/analysisrun/pull/51
- feat(analysisrun): scan_model_input中断時のメッセージを改善 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/52
- 標準入力から入力を受け取るPipeableRunnerの実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/57
- Pipeable by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/47
- chore: 不要になったスクリプトを削除しREADMEを更新 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/58
- feat(analysisrun): ユーザー入力にquestionaryを使用 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/59
- feat(anaiysisrun): "showschema"モードを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/64
- docs(analiysisrun): docstringを更新 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/61
- feat(analysisrun): scanner.scanを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/65
- fix(analysisrun): デフォルトで解析結果に追加するデータ名の列名を変更 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/66

## [v0.0.6](https://github.com/cosomil/analysisrun/compare/v0.0.5...v0.0.6) - 2025-06-16
- docs by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/37
- fix: マングリングエラーとpickleエラーを解消 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/39

## [v0.0.5](https://github.com/cosomil/analysisrun/compare/v0.0.4...v0.0.5) - 2025-06-13
- 数値解析の対象となるデータのエンティティ名(Entity列)を指定できるようにする by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/32
- 複数の画像解析結果CSVを解析対象として扱う by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/36

## [v0.0.4](https://github.com/cosomil/analysisrun/compare/v0.0.3...v0.0.4) - 2025-06-06
- docs: インストールコマンドを更新 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/26
- feat: `Runner`に各レーンの解析完了後のpostprocessを行う機能を追加 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/28
- docs: インストールコマンドを更新 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/30

## [v0.0.3](https://github.com/cosomil/analysisrun/compare/v0.0.2...v0.0.3) - 2025-06-05
- uvのインストールコマンドを変更 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/19
- ドキュメント更新 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/21
- chore: install.ps1のみBOMを削除 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/22
- docs: uvのインストールコマンドを修正 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/23
- chore: "視野"の英語表現を"field"に変更 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/25

## [v0.0.2](https://github.com/cosomil/analysisrun/compare/v0.0.1...v0.0.2) - 2025-05-19
- feat: ファイルパス、ディレクトリパスの入力を受けるための型を追加 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/17

## [v0.0.1](https://github.com/cosomil/analysisrun/commits/v0.0.1) - 2025-05-15
- NotebookRunnerを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/1
- Fix notebook runner by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/2
- feat: Pydanticのモデルの各フィールドの値を対話的に入力するscan_model_inputを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/3
- ParallelRunner by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/4
- スクリプト実行ヘルパーを実装 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/5
- ci: tagprを導入 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/6
- change: Outputをcallableに変更 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/13
- fix: デフォルト引数は都度生成する by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/14
- `scan**`メソッドでなく`__iter__`メソッドを実装する by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/15
- docs: コンテキストオブジェクトの推奨事項を削除 by @daichitakahashi in https://github.com/cosomil/analysisrun/pull/16
