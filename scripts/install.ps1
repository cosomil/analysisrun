# uvのプロジェクトとして作成されたPythonスクリプトを実行するPowerShellスクリプトのショートカットをデスクトップに作成する。

# 現在のスクリプトのディレクトリを取得
$currentDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# exec.ps1 のパスを設定
$scriptPath = Join-Path -Path $currentDir -ChildPath "exec.ps1"

# パスの検証
if (-not (Test-Path -Path $scriptPath -PathType Leaf)) {
    Write-Error "実行ファイル '$scriptPath' が存在しません。同じディレクトリにexec.ps1があることを確認してください。"
    exit 1
}

# デフォルトのショートカット名を設定
$defaultShortcutName = "Pythonスクリプトを実行"

# ユーザーからショートカット名を入力してもらう
Write-Host "ショートカット名を入力してください。（デフォルト: $defaultShortcutName）:"
$userInput = Read-Host

# 入力が空の場合はデフォルト値を使用
if ([string]::IsNullOrWhiteSpace($userInput)) {
    $shortcutName = $defaultShortcutName
} else {
    $shortcutName = $userInput
}

# デスクトップのパスを取得
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path -Path $desktopPath -ChildPath "$shortcutName.lnk"

# ショートカットを作成
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -NoProfile -File `"$scriptPath`""
    $shortcut.WorkingDirectory = [System.IO.Path]::GetDirectoryName($scriptPath)
    $shortcut.IconLocation = "powershell.exe,0"
    $shortcut.Description = "$shortcutName PowerShellスクリプトを実行"
    $shortcut.Save()
    Write-Host "ショートカットが正常に作成されました: $shortcutPath" -ForegroundColor Green
    Write-Host "デスクトップ上の '$shortcutName' アイコンをクリックしてスクリプトを実行できます。" -ForegroundColor Green
}
catch {
    Write-Error "ショートカットの作成中にエラーが発生しました: $_"
    exit 1
}
