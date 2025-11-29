param(
    [string]$EnvName = "vanishing-gradient-demo",
    [string]$PythonVersion = "3.11"
)

Write-Host "=================================="
Write-Host "Vanishing Gradient Demo - Conda Setup"
Write-Host "=================================="
Write-Host ""

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda command not found. Install Miniconda/Anaconda and restart PowerShell."
    exit 1
}

# Initialize conda for PowerShell if needed
& conda info | Out-Null

$envList = conda env list
$envExists = $envList -split "`n" | Where-Object { $_ -match "^$EnvName\s" }
if (-not $envExists) {
    Write-Host "Creating conda environment '$EnvName' with Python $PythonVersion..."
    conda create -y -n $EnvName "python=$PythonVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create environment"
        exit 1
    }
} else {
    Write-Host "Environment '$EnvName' already exists; reusing."
}

Write-Host ""
Write-Host "Activating conda environment..."
conda activate $EnvName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate environment"
    exit 1
}

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip"
    exit 1
}

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install requirements"
    exit 1
}

Write-Host ""
Write-Host "Frontend libraries load via CDN (Chart.js, Netron), so no extra npm install is required."
Write-Host ""
Write-Host "=================================="
Write-Host "✓ Conda environment ready!"
Write-Host "=================================="
Write-Host ""
Write-Host "To start developing in this session run:"
Write-Host "  conda activate $EnvName"
Write-Host "  python app.py"
Write-Host ""
Write-Host "Open http://localhost:5000 to view the UI."
