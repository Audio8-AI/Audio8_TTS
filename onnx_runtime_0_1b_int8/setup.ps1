$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = $env:PYTHON_BIN
$pythonArgs = @()

if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        $pythonExe = $pyLauncher.Source
        $pythonArgs = @("-3")
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            $pythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if ($null -ne $pythonCommand) {
            $pythonExe = $pythonCommand.Source
        }
    }
}

if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    throw "Python was not found. Install Python 3.10+ and add it to PATH."
}

& $pythonExe @pythonArgs -c "import sys; raise SystemExit(sys.version_info < (3, 10))"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 or newer is required."
}

$venv = Join-Path $Root ".venv"
& $pythonExe @pythonArgs -m venv $venv
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the virtual environment at $venv."
}

$venvPython = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "The virtual environment was created, but $venvPython was not found."
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $venvPython -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install runtime requirements." }

Write-Host "Environment ready: $venvPython"
