param(
    [int]$CameraId = 1,
    [string]$LaserPort = "COM6",
    [int]$LaserBaud = 115200,
    [string]$LogCsv = "",
    [switch]$Fp16,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$bundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $bundleDir "live_geometry_v2.py"
$candidates = @(
    $env:PALLET_POSE_PYTHON,
    (Join-Path $bundleDir "runtime\python.exe"),
    "C:\Users\DELL\anaconda3\envs\pallet-pose\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$pythonPath = $candidates | Select-Object -First 1

if (-not $pythonPath) {
    throw "Compatible Python was not found. Run setup_live_windows.cmd first."
}

$env:PYTHONDONTWRITEBYTECODE = "1"
& $pythonPath -c "import torch,torchvision,cv2,serial,yacs,skimage"
if ($LASTEXITCODE -ne 0) {
    throw "Required Python packages are missing from $pythonPath. Run setup_live_windows.cmd."
}

$liveArgs = @(
    $appPath,
    "--camera-id", $CameraId,
    "--laser-port", $LaserPort,
    "--laser-baud", $LaserBaud
)
if ($Fp16) {
    $liveArgs += "--fp16"
}
if ($LogCsv) {
    $liveArgs += @("--log-csv", $LogCsv)
}
if ($ExtraArgs) {
    $liveArgs += $ExtraArgs
}

& $pythonPath @liveArgs
exit $LASTEXITCODE
