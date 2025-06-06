# analysisrun
解析メソッドのベースとなるヘルパーを提供します。

## ライブラリ
### インストール方法
```shell
$ uv add git+https://github.com/cosomil/analysisrun --tag v0.0.4
```
※適切なバージョン（タグ）を選択してください。
※GitHubで公開されたパッケージをインストールするためには、システムにGitがインストールされている必要があります。

### 使用方法
[ドキュメント](documents.md)を参照。

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
