# One-Click Pallet Recognition Launch Script
# Run this from PowerShell (Administrator)

Write-Host "--- Pallet Recognition System (V2) ---" -ForegroundColor Cyan

# 1. Hardware Initialization
Write-Host "[1/3] Searching for RealSense D435i..." -ForegroundColor Yellow
$rs = (usbipd list | Select-String "RealSense")
if ($rs) {
    $busid = $rs.ToString().Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)[0]
    Write-Host "Found Camera at BUSID: $busid. Attaching to WSL..." -ForegroundColor Green
    usbipd attach --wsl --busid $busid
} else {
    Write-Host "Error: Camera not found. Please check physical connection." -ForegroundColor Red
    exit
}

# 2. Build Environment (if needed)
Write-Host "[2/3] Preparing Docker environment..." -ForegroundColor Yellow
docker-compose build

# 3. Launch Inference
Write-Host "[3/3] Launching Pallet Recognition! (Ctrl+C to quit)" -ForegroundColor Cyan
Write-Host "Make sure VcXsrv is running with 'Disable access control' checked." -ForegroundColor Gray
docker-compose up
