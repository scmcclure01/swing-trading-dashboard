#!/bin/bash
# Swing Trading Screener — Weekly Runner
# Scheduled to run every Sunday at 7:00 AM

SCRIPT_DIR="/Users/scotmcclure/Library/CloudStorage/GoogleDrive-scmcclure01@gmail.com/My Drive/Claude/Stock Trading/Swing Trading Framework"
LOG_FILE="$SCRIPT_DIR/screener_log.txt"
PYTHON="/Library/Developer/CommandLineTools/usr/bin/python3"

echo "========================================" >> "$LOG_FILE"
echo "Run started: $(date)" >> "$LOG_FILE"

cd "$SCRIPT_DIR" && "$PYTHON" screener.py >> "$LOG_FILE" 2>&1

echo "Run finished: $(date)" >> "$LOG_FILE"
