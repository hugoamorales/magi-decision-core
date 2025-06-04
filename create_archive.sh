#!/bin/bash

# Script: create_archive.sh
# Purpose: Creates a zip archive of the MAGI Decision Core v5 project files for distribution.
# Usage: ./create_archive.sh
# Output: Generates magi_decision_core_v5.zip in the current directory.

# Create a directory for the files
mkdir -p magi_decision_core_v5

# List of files to include in the archive
files=(
    "magi_system_v5.py"
    "test_magi.py"
    "setup.py"
    "requirements.txt"
    "config.json"
    "README.md"
    # ".env"  # Uncomment to include .env (not recommended for sharing)
)

# Check if files exist before copying
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found in current directory"
        exit 1
    fi
    cp "$file" "magi_decision_core_v5/"
done

# Create zip archive of the directory
zip -r magi_decision_core_v5.zip magi_decision_core_v5

# Clean up temporary directory
rm -r magi_decision_core_v5

echo "Created magi_decision_core_v5.zip"