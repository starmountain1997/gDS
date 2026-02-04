#!/bin/bash

# Source the virtual environment
source .venv/bin/activate

# Run the Python script to save GitHub logs
python save_gh_logs.py

# Get current timestamp for the commit message
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Add all changes to git
git add logs/

# Create a commit
git commit -m "track ci data + $TIMESTAMP"

echo "Script finished. Review the commit and push if desired."
