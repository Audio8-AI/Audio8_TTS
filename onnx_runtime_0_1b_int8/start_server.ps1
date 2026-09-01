$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual environment not found. Run setup.ps1 first."
}

$port = if ([string]::IsNullOrWhiteSpace($env:PORT)) { "8024" } else { $env:PORT }
$pidFile = Join-Path $Root ".service.pid"
if (Test-Path -LiteralPath $pidFile) {
    $existingId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    if (Get-Process -Id $existingId -ErrorAction SilentlyContinue) {
        Write-Host "Service is already running with PID $existingId"
        exit 0
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

if ([string]::IsNullOrWhiteSpace($env:ARKTTS_MODEL_DIR)) { $env:ARKTTS_MODEL_DIR = Join-Path $Root "model" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_VOICES_DIR)) { $env:ARKTTS_VOICES_DIR = Join-Path $Root "voices" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_REGISTRATION_DIR)) { $env:ARKTTS_REGISTRATION_DIR = Join-Path $env:ARKTTS_MODEL_DIR "registration" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_PRECISION)) { $env:ARKTTS_PRECISION = "int8" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_CODEC_PRECISION)) { $env:ARKTTS_CODEC_PRECISION = "fp16" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_THREADS)) { $env:ARKTTS_THREADS = "5" }

$logPath = Join-Path $Root "service.log"
$errorPath = Join-Path $Root "service.error.log"
$hostName = if ([string]::IsNullOrWhiteSpace($env:HOST)) { "127.0.0.1" } else { $env:HOST }
$quotedRoot = '"' + $Root.Replace('"', '\"') + '"'
$arguments = "-m uvicorn arktts_runtime.service:app --app-dir $quotedRoot --host $hostName --port $port"
$process = Start-Process -FilePath $venvPython -ArgumentList $arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $logPath -RedirectStandardError $errorPath -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "Audio8 0.1B TTS is ready: http://127.0.0.1:$port"
            exit 0
        }
    } catch {
        # The service can take time to load the ONNX sessions.
    }
    if ($process.HasExited) {
        throw "Service exited during startup. See $logPath and $errorPath"
    }
    Start-Sleep -Seconds 1
}

throw "Service did not become ready in 60 seconds. See $logPath and $errorPath"
