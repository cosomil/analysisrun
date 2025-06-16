# analysisrun
解析メソッドのベースとなるヘルパーを提供します。

## ライブラリ
### インストール方法
```shell
$ uv add git+https://github.com/cosomil/analysisrun --tag v0.0.6
```
※適切なバージョン（タグ）を選択してください。
※GitHubで公開されたパッケージをインストールするためには、システムにGitがインストールされている必要があります。

### 使用方法
```python
import pandas as pd

from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs
from analysisrun.cleansing import filter_by_entity

main_data = pd.read_csv("./path/to/the/file.csv")
enhance_data = pd.read_csv("./path/to/the/file_enhancement.csv")

class Context:
    your_analysis_parameter: str

# 各レーンの解析を行う関数
def analyze(args: AnalysisArgs[Context]):
    ctx, output = args.ctx, ctx.output
    fields, [additional_fields] = args.fields, args.fields_for_enhancement

    for (field, additional_field) in zip(fields, additional_fields):
        # ある視野の情報と追加データの同じ視野の情報を用いて解析処理を実行
        # ...

    # ...

    # 画像の出力
    output(fig, "filename.png", "result_graph")

    # ...

    return pd.Series({ ... })

# 全ての解析結果をマージしたDataFrameに対し、さらに後処理を加えるための関数(optional)
def postprocess(args: PostprocessArgs[Context]):
    results = args.analysis_results
    if False in results.ok:
      results[ok] = False # 1件でも結果不良のデータがあれば全て不可と判定する
    return results

runner = ParallelRunner(analyze, postprocess)
result = runner.run(
  ctx=Context(your_analysis_parameter="aaaaaa"),
  target_data=["A", "B", "C"],
  whole_data=filter_by_entity(main_data, entity="Activity Spots"),
  data_for_enhancement=[filter_by_entity(enhance_data, entity="Total Count")],
)

result.to_csv("path/to/result/file.csv")

```

各モジュールの詳細は[ドキュメント](documents.md)を参照。

## Pythonスクリプト実行ヘルパー
### 必要条件
- PowerShellがインストールされている
- Gitがインストールされている
  - 参考: https://qiita.com/takeru-hirai/items/4fbe6593d42f9a844b1c
- uvがインストールされている
  - `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - [uv(Pythonのパッケージマネージャ)の使い方](https://www.notion.so/cosomil/uv-Python-200ac7552b3c80f19190e94a67daf175)

### インストール方法
PowerShellを開き、以下のコマンドを貼り付けて実行してください。
```ps
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/cosomil/analysisrun/refs/heads/main/scripts/install.ps1 | iex"
```
成功すると、デスクトップに「Pythonスクリプトを実行する」というショートカットが作成されます。

### 使用方法
ショートカットに、実行したいPythonスクリプトが入ったフォルダをドラッグ＆ドロップしてください。

- Pythonスクリプトは`uv`で作成されたプロジェクトである必要があり、依存関係を自動的にインストールします
- デフォルトではフォルダーに含まれる`main.py`を実行します。存在しない場合は他のスクリプトを実行することができます
