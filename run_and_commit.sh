#!/bin/bash

# Source the virtual environment
conda activate gDS

# Run the Python script to save GitHub logs
python3 save_gh_logs.py

# Get current timestamp for the commit message
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Add all changes to git
git add logs/

# Create a commit
git commit -m "track ci data $TIMESTAMP"

git push

echo "Script finished. Changes committed and pushed."
