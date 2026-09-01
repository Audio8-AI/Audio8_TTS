$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual environment not found. Run setup.ps1 first."
}

$modelDir = if ([string]::IsNullOrWhiteSpace($env:ARKTTS_MODEL_DIR)) {
    Join-Path $Root "model"
} else { $env:ARKTTS_MODEL_DIR }
$voicesDir = if ([string]::IsNullOrWhiteSpace($env:ARKTTS_VOICES_DIR)) {
    Join-Path $Root "voices"
} else { $env:ARKTTS_VOICES_DIR }

Push-Location $Root
try {
    & $venvPython -m arktts_runtime.cli --model-dir $modelDir --voices-dir $voicesDir @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
