param([switch]$NoBrowser, [int]$Port = 5088)
$ErrorActionPreference = 'Stop'
$hostDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$python = if ($env:PIPER_PYTHON) { $env:PIPER_PYTHON } else { Join-Path $env:USERPROFILE 'anaconda3\envs\piper_act\python.exe' }
$web = Join-Path $hostDir 'web_app.py'
$url = "http://127.0.0.1:$Port"
$started = $null

function Get-ConsoleHealth {
    try { return Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2 } catch { return $null }
}
function Open-Console {
    if (-not $NoBrowser) { Start-Process -FilePath "$url/#capture" }
}
try {
    if ($Port -lt 1 -or $Port -gt 65535) { throw '端口必须在 1–65535 之间。' }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "未找到 Python：$python" }
    if (-not (Test-Path -LiteralPath $web -PathType Leaf)) { throw "未找到上位机后端：$web" }
    $health = Get-ConsoleHealth
    if ($health -and $health.application -eq 'piper-act-console' -and $health.root -eq $hostDir) {
        Write-Host "上位机已运行：$url （PID $($health.pid)）"
        Open-Console
        exit 0
    }
    # Never open an unrelated application or overwrite its listener.
    $probe = New-Object System.Net.Sockets.TcpClient
    try {
        $connected = $probe.ConnectAsync('127.0.0.1', $Port)
        try { $null = $connected.Wait(700) } catch { }
        if ($probe.Connected) { throw "端口 $Port 已被其他或旧版服务占用，请先关闭该服务，或使用 -Port 指定其他端口。" }
    } finally { $probe.Dispose() }

    $front = Join-Path $hostDir 'frontend'
    $index = Join-Path $front 'dist\index.html'
    $buildNeeded = -not (Test-Path -LiteralPath $index)
    if (-not $buildNeeded) {
        $built = (Get-Item -LiteralPath $index).LastWriteTimeUtc
        $sources = @(Get-ChildItem -LiteralPath (Join-Path $front 'src') -Recurse -File)
        $sources += Get-Item -LiteralPath (Join-Path $front 'index.html'), (Join-Path $front 'review.html'), (Join-Path $front 'package.json'), (Join-Path $front 'package-lock.json'), (Join-Path $front 'vite.config.js')
        $buildNeeded = [bool]($sources | Where-Object { $_.LastWriteTimeUtc -gt $built } | Select-Object -First 1)
    }
    if ($buildNeeded) {
        if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw '前端需要构建，但未找到 npm。请安装 Node.js 后重试。' }
        if (-not (Test-Path -LiteralPath (Join-Path $front 'node_modules\vite'))) { throw "请先在 $front 运行 npm ci，再双击启动。" }
        Write-Host '正在构建更新后的前端…'
        Push-Location -LiteralPath $front
        try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw '前端构建失败。' } } finally { Pop-Location }
    }
    $logDir = Join-Path $hostDir 'Log'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $outLog = Join-Path $logDir "console-$stamp.out.log"
    $errLog = Join-Path $logDir "console-$stamp.err.log"
    $runtimePrefix = Split-Path -Parent $python
    $env:PATH = "$runtimePrefix;$runtimePrefix\Library\mingw-w64\bin;$runtimePrefix\Library\usr\bin;$runtimePrefix\Library\bin;$runtimePrefix\Scripts;$env:PATH"
    $env:CONDA_PREFIX = $runtimePrefix
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $arguments = @('-u', ('"' + $web + '"'), '--host', '127.0.0.1', '--port', "$Port")
    $started = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $hostDir -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    Write-Host '正在启动上位机服务…'
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    do {
        $started.Refresh()
        if ($started.HasExited) {
            $detail = Get-Content -LiteralPath $errLog -Encoding UTF8 -Tail 25 | Out-String
            throw "后端启动失败（退出码 $($started.ExitCode)）。`n$detail`n日志：$errLog"
        }
        $health = Get-ConsoleHealth
        if ($health -and $health.application -eq 'piper-act-console' -and $health.root -eq $hostDir -and $health.pid -eq $started.Id) {
            Write-Host "启动成功：$url"
            Write-Host "后台进程：$($started.Id)；日志：$logDir"
            Open-Console
            exit 0
        }
        Start-Sleep -Milliseconds 300
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "启动超时，请查看日志：$errLog"
} catch {
    # Only terminate the process launched by this invocation on startup failure.
    if ($started -and -not $started.HasExited) { Stop-Process -Id $started.Id -ErrorAction SilentlyContinue }
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
