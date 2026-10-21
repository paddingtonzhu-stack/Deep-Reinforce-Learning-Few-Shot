param()

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".vizdoom-play"

function Find-Python {
    $commands = @("py", "python")
    foreach ($command in $commands) {
        $resolved = Get-Command $command -ErrorAction SilentlyContinue
        if ($resolved) {
            try {
                & $resolved.Source --version 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    return $resolved.Source
                }
            } catch {}
        }
    }
    throw "Python 3 was not found. Install Python from python.org, enable 'Add Python to PATH', then rerun this script."
}

$BootstrapPython = Find-Python
if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating the Windows player environment..."
    & $BootstrapPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $VenvDir" }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
Write-Host "Installing/updating ViZDoom..."
& $Python -m pip install --upgrade pip vizdoom
if ($LASTEXITCODE -ne 0) { throw "ViZDoom installation failed." }

& $Python -c "import vizdoom; print('ViZDoom player ready:', vizdoom.__version__)"
if ($LASTEXITCODE -ne 0) { throw "ViZDoom import check failed." }

Write-Host "Setup complete. Play with: .\play_battle.ps1 -Map 1"
