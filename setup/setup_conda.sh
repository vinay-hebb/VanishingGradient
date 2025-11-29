#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_conda.sh [env-name]
# Creates a conda environment, installs Python dependencies, and prepares UI assets.

ENV_NAME="${1:-vanishing-gradient-demo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "=================================="
echo "Vanishing Gradient Demo - Conda Setup"
echo "=================================="
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found on your PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists; skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo ""
echo "Activating environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo ""
echo "Upgrading pip and installing backend/UI Python requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Frontend assets are delivered via CDN (Chart.js, Netron), so no extra node/npm steps are needed."

echo ""
echo "=================================="
echo "✓ Conda environment ready!"
echo "=================================="
echo ""
echo "To start working run:"
echo "  conda activate ${ENV_NAME}"
echo "  python app.py"
echo ""
echo "App will be available at http://localhost:5000"
echo ""
