param(
    [ValidateSet("deadly_corridor", "battle1", "battle2")]
    [string]$Level = "deadly_corridor",
    [ValidateSet("spectator", "random", "noop")]
    [string]$Mode = "spectator",
    [int]$Episodes = 1,
    [int]$Seed = 12345
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$PythonCandidates = @(
    (Join-Path $ProjectDir ".venv\Scripts\python.exe"),
    (Join-Path $ProjectDir ".python-corridor\python.exe"),
    (Join-Path $ProjectDir ".python\python.exe"),
    (Join-Path $ProjectDir ".sf\Scripts\python.exe"),
    (Join-Path $ProjectDir ".sf\bin\python")
)

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path -LiteralPath $Candidate) {
        try {
            & $Candidate -c "import vizdoom" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $Python = $Candidate
                break
            }
        } catch {}
    }
}
if (-not $Python) {
    throw "No working .venv with ViZDoom found. Run .\setup_vizdoom_player.ps1 once, then retry."
}

& $Python inspect_vizdoom_level.py --level $Level --mode $Mode --episodes $Episodes --seed $Seed
