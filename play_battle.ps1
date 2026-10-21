param(
    [ValidateSet(1, 2)]
    [int]$Map = 1,
    [int]$Episodes = 3,
    [int]$Seed = 12345,
    [switch]$Mute
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Viewer = Join-Path $ProjectDir "inspect_vizdoom_level.ps1"
$Level = "battle$Map"

if ($Mute) {
    # The generic launcher currently has no switch forwarding, so invoke its
    # selected player directly for muted sessions.
    $Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Run .\setup_vizdoom_player.ps1 first."
    }
    & $Python (Join-Path $ProjectDir "inspect_vizdoom_level.py") --level $Level --mode spectator --episodes $Episodes --seed $Seed --mute
} else {
    & $Viewer -Level $Level -Mode spectator -Episodes $Episodes -Seed $Seed
}
