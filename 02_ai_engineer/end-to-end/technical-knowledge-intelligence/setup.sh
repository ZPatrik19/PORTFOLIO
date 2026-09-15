#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo " TECHNICAL KNOWLEDGE INTELLIGENCE - FIRST SETUP"
echo "================================================================"
echo
echo "This script prepares the local environment and starts the app."
echo "After the first successful setup, use ./run_project.sh for daily use."
echo

./run_project.sh setup
echo
echo "[INFO] Setup finished. Starting the application..."
./run_project.sh run
