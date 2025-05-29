# uvのプロジェクトとして作成されたPythonスクリプトを実行するPowerShellスクリプトのショートカットをデスクトップに作成する。

Add-Type -AssemblyName System.Windows.Forms

$folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
$folderDialog.Description = "スクリプトを保存するフォルダーを選択してください"
if ($folderDialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "キャンセルされました"
    exit
}
$targetFolder = $folderDialog.SelectedPath

# exec.ps1スクリプトを指定されたフォルダーにダウンロード
$scriptUrl = "https://raw.githubusercontent.com/cosomil/analysisrun/refs/heads/main/scripts/exec.ps1"
$scriptName = "exec.ps1"
$scriptPath = Join-Path $targetFolder $scriptName
Invoke-WebRequest -Uri $scriptUrl -OutFile $scriptPath

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
