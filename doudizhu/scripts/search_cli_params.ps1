param(
    [string]$PythonBin = "",
    [string]$Gpu = "0",
    [int]$TrialSeconds = 90,
    [int]$CooldownSeconds = 3,
    [int]$Repeats = 2,
    [string]$SaveDir = "search_outputs"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($PythonBin)) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $py) {
        $py = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if ($null -eq $py) {
        throw "No python interpreter found in PATH."
    }
    $PythonBin = $py.Source
}

$Args = @(
    "scripts/search_cli_params.py",
    "--python", $PythonBin,
    "--gpu", $Gpu,
    "--search-mode", "two-stage",
    "--trial-seconds", "$TrialSeconds",
    "--cooldown-seconds", "$CooldownSeconds",
    "--repeats", "$Repeats",
    "--batch-sizes", "8,16,32",
    "--unroll-lengths", "8,16",
    "--replay-warmups", "4,8",
    "--replay-sizes", "64,128",
    "--savedir", $SaveDir
)

Write-Host "Repo root :" $RepoRoot
Write-Host "Python    :" $PythonBin
Write-Host ""
Write-Host "$PythonBin $($Args -join ' ')"
Write-Host ""

Push-Location $RepoRoot
try {
    & $PythonBin @Args
}
finally {
    Pop-Location
}
