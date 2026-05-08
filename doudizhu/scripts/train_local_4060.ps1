param(
    [string]$PythonBin = "",
    [string]$Gpu = "0",
    [string]$Xpid = "local_4060_run1",
    [string]$SaveDir = "local_runs",
    [int]$TotalFrames = 1000000
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
    "train.py",
    "--xpid", $Xpid,
    "--savedir", $SaveDir,
    "--actor_device_cpu",
    "--gpu_devices", $Gpu,
    "--training_device", $Gpu,
    "--num_actor_devices", "1",
    "--num_actors", "2",
    "--num_threads", "2",
    "--batch_size", "16",
    "--unroll_length", "16",
    "--replay_buffer_size", "64",
    "--replay_warmup_size", "8",
    "--total_frames", "$TotalFrames",
    "--save_interval", "1000"
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
