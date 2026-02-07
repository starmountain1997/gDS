#!/bin/bash

# Source the virtual environment
conda activate gDS

# Run the Python script to save GitHub logs
SCRIPT_DIR="$(dirname "$0")"
python3 "$SCRIPT_DIR/save_gh_logs.py"

# Get current timestamp for the commit message
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Add all changes to git
git add gh_action_watcher/logs/

# Create a commit
git commit -m "track ci data $TIMESTAMP"

git push

echo "Script finished. Changes committed and pushed."
