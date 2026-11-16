[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$logStarted = $false
try {
    $config = Join-Path $PSScriptRoot 'LOCAL_REPOSITORY.txt'
    if (!(Test-Path -LiteralPath $config -PathType Leaf)) {
        throw '找不到 LOCAL_REPOSITORY.txt，请把启动文件和配置文件放在更新包同一个文件夹。'
    }
    $repo = (Get-Content -LiteralPath $config -Raw -Encoding UTF8).Trim().Trim('"')
    if ($repo -match 'https?://|git@github\.com') {
        throw '配置里需要填写电脑上的文件夹路径，不能填写 GitHub 网页地址。'
    }
    if (!(Test-Path -LiteralPath (Join-Path $repo '.git'))) {
        throw ('本地仓库不存在：' + $repo + '。请检查该文件夹是否被移动或删除。')
    }
    $repo = (Resolve-Path -LiteralPath $repo).Path
    $updater = Join-Path $PSScriptRoot 'Update-HGV.ps1'
    foreach ($name in @('Update-HGV.ps1','UPDATE_MANIFEST.tsv','UPDATE_METADATA.json','payload')) {
        if (!(Test-Path -LiteralPath (Join-Path $PSScriptRoot $name))) {
            throw ('更新包缺少 ' + $name + '，请保留完整的原更新包。')
        }
    }
    $logDir = Join-Path $PSScriptRoot 'logs'
    [IO.Directory]::CreateDirectory($logDir) | Out-Null
    $logPath = Join-Path $logDir ('one_click_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
    Start-Transcript -LiteralPath $logPath | Out-Null
    $logStarted = $true
    Write-Host ''
    Write-Host 'HGV 一键更新' -ForegroundColor Cyan
    Write-Host ('已自动找到本地仓库：' + $repo)
    if ($CheckOnly) {
        Write-Host '本次仅检查，不复制、不提交、不上传。'
        $steps = @('Plan')
    } else {
        Write-Host '将自动检查、复制更新文件，然后提交并上传 GitHub。'
        Write-Host '如果 GitHub 弹出登录窗口，请用你自己的账号完成登录。'
        $steps = @('Plan','Apply','Publish')
    }
    $labels = @{ Plan='检查文件和哈希'; Apply='复制到本地仓库'; Publish='提交、上传并核对 GitHub' }
    $i = 0
    foreach ($step in $steps) {
        $i++
        Write-Host ''
        Write-Host ('[' + $i + '/' + $steps.Count + '] ' + $labels[$step] + '，请稍等……') -ForegroundColor Cyan
        # Each stage is a separate process because the audited updater uses exit codes.
        & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File $updater -RepoPath $repo -Mode $step
        if ($LASTEXITCODE -ne 0) {
            throw ('“' + $labels[$step] + '”未完成。已停止后续步骤，请查看上方 STOP 信息。')
        }
    }
    Write-Host ''
    if ($CheckOnly) {
        Write-Host '检查通过，本地仓库和 GitHub 均未改动。' -ForegroundColor Green
    } else {
        Write-Host '更新完成：已上传 GitHub，并核对远端 commit。' -ForegroundColor Green
        Write-Host 'https://github.com/nhjie-bio/HGV-reference-screening'
    }
    Write-Host ('运行记录：' + $logPath)
    Stop-Transcript | Out-Null
    $logStarted = $false
    exit 0
} catch {
    Write-Host ''
    Write-Host ('没有完成：' + $_.Exception.Message) -ForegroundColor Red
    Write-Host '请保留窗口中的报错；已复制的文件会在下次运行时核对并跳过。'
    if ($logStarted) { Stop-Transcript | Out-Null }
    exit 1
}
