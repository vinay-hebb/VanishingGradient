Write-Host "=================================="
Write-Host "Vanishing Gradient Demo - Setup"
Write-Host "=================================="
Write-Host ""

try {
    $pythonVersion = (python --version)
    Write-Host "Detected $pythonVersion"
} catch {
    Write-Error "Python is not available. Install Python 3.11+ and ensure 'python' is on PATH."
    exit 1
}

Write-Host ""
Write-Host "Upgrading pip and installing dependencies..."
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
Write-Host "=================================="
Write-Host "✓ Setup completed successfully!"
Write-Host "=================================="
Write-Host ""
Write-Host "To run the application:"
Write-Host "  python app.py"
Write-Host ""
Write-Host "Then browse to http://localhost:5000"
