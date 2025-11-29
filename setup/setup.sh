#!/bin/bash

echo "=================================="
echo "Vanishing Gradient Demo - Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version || { echo "Python not found! Please install Python 3.11+"; exit 1; }

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install -q --upgrade pip
pip install -q -r requirements.txt || { echo "Failed to install Python packages"; exit 1; }

echo ""
echo "=================================="
echo "✓ Setup completed successfully!"
echo "=================================="
echo ""
echo "To run the application:"
echo "  python app.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:5000"
echo ""
