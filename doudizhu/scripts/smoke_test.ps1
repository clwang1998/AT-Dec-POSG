param(
    [switch]$Cpu,
    [string]$Gpu = "0",
    [string]$PythonBin = "",
    [string]$Xpid = "",
    [string]$SaveDir = "",
    [int]$TotalFrames = 10
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

if ([string]::IsNullOrWhiteSpace($SaveDir)) {
    $SaveDir = Join-Path $RepoRoot "smoke_outputs"
}

$Mode = if ($Cpu) { "cpu" } else { "gpu" }
if ([string]::IsNullOrWhiteSpace($Xpid)) {
    $Xpid = "smoke_${Mode}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

Write-Host "Repo root :" $RepoRoot
Write-Host "Python    :" $PythonBin
& $PythonBin -V

Write-Host "Torch env :"
& $PythonBin -c @"
import sys
try:
    import torch
    print(f"  executable={sys.executable}")
    print(f"  torch={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device_count={torch.cuda.device_count()}")
        print(f"  current_device={torch.cuda.current_device()}")
        print(f"  device_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"  torch import failed: {exc}")
"@

$Args = @(
    "train.py",
    "--num_actor_devices", "1",
    "--num_actors", "1",
    "--num_threads", "1",
    "--batch_size", "1",
    "--unroll_length", "1",
    "--replay_buffer_size", "2",
    "--replay_warmup_size", "1",
    "--total_frames", "$TotalFrames",
    "--save_interval", "1000",
    "--xpid", $Xpid,
    "--savedir", $SaveDir
)

if ($Cpu) {
    $Args += @("--actor_device_cpu", "--training_device", "cpu", "--gpu_devices", "")
} else {
    $Args += @("--actor_device_cpu", "--training_device", $Gpu, "--gpu_devices", $Gpu)
}

Write-Host ""
Write-Host "Running $Mode smoke test..."
Write-Host "$PythonBin $($Args -join ' ')"
Write-Host ""

Push-Location $RepoRoot
try {
    & $PythonBin @Args
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Smoke test finished."
Write-Host "Artifacts:" (Join-Path $SaveDir $Xpid)
