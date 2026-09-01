$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual environment not found. Run setup.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($env:ARKTTS_MODEL_DIR)) { $env:ARKTTS_MODEL_DIR = Join-Path $Root "model" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_VOICES_DIR)) { $env:ARKTTS_VOICES_DIR = Join-Path $Root "voices" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_REGISTRATION_DIR)) { $env:ARKTTS_REGISTRATION_DIR = Join-Path $env:ARKTTS_MODEL_DIR "registration" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_PRECISION)) { $env:ARKTTS_PRECISION = "int8" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_CODEC_PRECISION)) { $env:ARKTTS_CODEC_PRECISION = "fp16" }
if ([string]::IsNullOrWhiteSpace($env:ARKTTS_THREADS)) { $env:ARKTTS_THREADS = "5" }

$hostName = if ([string]::IsNullOrWhiteSpace($env:HOST)) { "127.0.0.1" } else { $env:HOST }
$port = if ([string]::IsNullOrWhiteSpace($env:PORT)) { "8024" } else { $env:PORT }

Push-Location $Root
try {
    & $venvPython -m uvicorn arktts_runtime.service:app --app-dir $Root --host $hostName --port $port @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
