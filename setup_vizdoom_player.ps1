param()

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"

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
    return $null
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BootstrapPython = $null
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BootstrapPython = Find-Python
}
if ((-not (Test-Path -LiteralPath $VenvPython)) -and (-not $BootstrapPython)) {
    throw "Cannot create .venv because no base Python interpreter is available. This script never installs Python system-wide."
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating isolated project environment: $VenvDir"
    & $BootstrapPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $VenvDir" }
}

$Python = $VenvPython
Write-Host "Using isolated environment: $VenvDir"
Write-Host "Installing/updating ViZDoom..."
& $Python -m pip install --upgrade pip vizdoom
if ($LASTEXITCODE -ne 0) { throw "ViZDoom installation failed." }

& $Python -c "import vizdoom; print('ViZDoom player ready:', vizdoom.__version__)"
if ($LASTEXITCODE -ne 0) { throw "ViZDoom import check failed." }

Write-Host "Setup complete. ViZDoom was installed only inside .venv."
Write-Host "Play with: .\play_battle.ps1 -Map 1"
