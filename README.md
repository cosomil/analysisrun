# analysisrun
解析メソッドのベースとなるヘルパーを提供します。

## ライブラリ
### インストール方法
```shell
$ uv add git+https://github.com/cosomil/analysisrun --tag 0.0.1
```
※適切なタグを選択してください。

### 使用方法
[ドキュメント](documents.md)を参照。

## Pythonスクリプト実行ヘルパー
### 必要条件
- PowerShellがインストールされている
- uvがインストールされている
  - `powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"`

### インストール方法
`scripts/install.ps1`を実行すると、デスクトップに「Pythonスクリプトを実行する」というショートカットを作成します。

### 使用方法
ショートカットに、実行したいPythonスクリプトが入ったフォルダをドラッグ＆ドロップしてください。

- Pythonスクリプトは`uv`で作成されたプロジェクトである必要があり、依存関係を自動的にインストールします
- デフォルトではフォルダーに含まれる`main.py`を実行します。存在しない場合は他のスクリプトを実行することができます
