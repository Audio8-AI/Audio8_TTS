$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $Root ".service.pid"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No managed Audio8 TTS service is running."
    exit 0
}

$processId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue
if ($null -ne $process) {
    Stop-Process -Id $processId
    try { Wait-Process -Id $processId -Timeout 30 -ErrorAction Stop } catch { }
}
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "Audio8 TTS service stopped."
