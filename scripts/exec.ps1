param(
    [Parameter(Position=0, Mandatory=$true)]
    [string]$FolderPath
)

# 指定されたパスが存在するかチェック
if (-not (Test-Path -Path $FolderPath)) {
    Write-Error "エラー: 指定されたパス '$FolderPath' が存在しません。"
    exit 1
}

# 指定されたパスがディレクトリであるかチェック
if (-not (Test-Path -Path $FolderPath -PathType Container)) {
    Write-Error "エラー: 指定されたパス '$FolderPath' はフォルダーではありません。"
    exit 1
}

# 指定されたフォルダーに移動
Set-Location -Path $FolderPath

# pyproject.tomlの存在確認
if (-not (Test-Path -Path "pyproject.toml" -PathType Leaf)) {
    Write-Error "エラー: このフォルダーでスクリプトを実行することはできません。pyproject.tomlファイルが存在するフォルダーを指定してください。"
    exit 1
}

# 実行対象のスクリプトを格納する変数（初期値はmain.py）
$scriptToRun = "main.py"

# main.pyが存在するかチェック
if (-not (Test-Path -Path $scriptToRun -PathType Leaf)) {
    # main.pyが存在しない場合、代替のPythonスクリプトを探す
    $pythonFiles = Get-ChildItem -Path . -Filter "*.py" -File
    
    if ($pythonFiles.Count -eq 0) {
        Write-Error "エラー: フォルダー内にPythonスクリプト(.py)が見つかりませんでした。"
        exit 1
    }
    elseif ($pythonFiles.Count -eq 1) {
        # スクリプトが1つしかない場合
        Write-Host "main.pyは見つかりませんでしたが、代わりに $($pythonFiles[0].Name) を実行します。"
        $scriptToRun = $pythonFiles[0].Name
    }
    else {
        # 複数のスクリプトがある場合、選択肢を表示
        Write-Host "main.pyは見つかりませんでした。実行するPythonスクリプトを選択してください:" -ForegroundColor Yellow
        
        for ($i = 0; $i -lt $pythonFiles.Count; $i++) {
            Write-Host "[$i] $($pythonFiles[$i].Name)"
        }
        
        $selection = Read-Host "番号を入力してください"
        
        if ($selection -match '^\d+$' -and [int]$selection -ge 0 -and [int]$selection -lt $pythonFiles.Count) {
            $scriptToRun = $pythonFiles[[int]$selection].Name
            Write-Host "スクリプト '$scriptToRun' を実行します..." -ForegroundColor Green
        }
        else {
            Write-Error "エラー: 無効な選択です。"
            exit 1
        }
    }
}

# 選択されたスクリプトを実行
uv run $scriptToRun

# 実行完了メッセージを表示
Write-Host "`n実行完了しました" -ForegroundColor Green
Read-Host "Enterキーを押すと終了します"
