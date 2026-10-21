param()

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExistingVenv = Join-Path $ProjectDir ".venv"
$PlayerVenv = Join-Path $ProjectDir ".vizdoom-play"
if (Test-Path -LiteralPath (Join-Path $ExistingVenv "Scripts\python.exe")) {
    $VenvDir = $ExistingVenv
    Write-Host "Using existing project environment: $VenvDir"
} else {
    $VenvDir = $PlayerVenv
}

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
    $Winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python 3 and winget were not found. Install Python 3.12 from python.org, then rerun this script."
    }
    Write-Host "Python 3 was not found. Installing Python 3.12 for the current Windows user..."
    & $Winget.Source install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python installation through winget failed." }

    $InstalledCandidates = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe")
    )
    foreach ($Candidate in $InstalledCandidates) {
        if (Test-Path -LiteralPath $Candidate) {
            $BootstrapPython = $Candidate
            break
        }
    }
    if (-not $BootstrapPython) {
        $BootstrapPython = Find-Python
    }
    if (-not $BootstrapPython) {
        throw "Python was installed but could not be located. Open a new PowerShell window and rerun this script."
    }
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating the Windows player environment..."
    & $BootstrapPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $VenvDir" }
}

$Python = $VenvPython
Write-Host "Installing/updating ViZDoom..."
& $Python -m pip install --upgrade pip vizdoom
if ($LASTEXITCODE -ne 0) { throw "ViZDoom installation failed." }

& $Python -c "import vizdoom; print('ViZDoom player ready:', vizdoom.__version__)"
if ($LASTEXITCODE -ne 0) { throw "ViZDoom import check failed." }

Write-Host "Setup complete. Play with: .\play_battle.ps1 -Map 1"
