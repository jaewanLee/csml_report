# Suggested Commands for BTC Prediction Project

## Environment Setup
```bash
# Create and activate conda environment
conda create -n csml python=3.13
conda activate csml

# Install dependencies
pip install -r requirements.txt
```

## Development Commands
```bash
# Code formatting
black .

# Type checking
pyright

# Testing
pytest

# Run data collection
python data_collection/scripts/btc_collector.py
```

## Git Commands
```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "message"

# Push
git push origin main
```

## System Commands (Darwin/macOS)
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search in files
grep -r "pattern" .

# Change directory
cd /path/to/directory
```