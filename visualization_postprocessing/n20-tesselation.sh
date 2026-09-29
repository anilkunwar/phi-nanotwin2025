#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=== Step 1: Configuring ~/.neperrc ==="
NEPERRC="$HOME/.neperrc"
CONFIG_LINE="neper -V -imagesize 800:400"

# Add the image size configuration if it doesn't already exist
if [ -f "$NEPERRC" ] && grep -Fxq "$CONFIG_LINE" "$NEPERRC"; then
    echo "Configuration already present in $NEPERRC"
else
    echo "$CONFIG_LINE" >> "$NEPERRC"
    echo "Added image size configuration to $NEPERRC"
fi

echo "=== Step 2: Generating 20-cell grain-growth tessellation ==="
# This creates n20-id1.tess
neper -T -n 20 -morpho gg

echo "=== Step 3: Visualizing and printing image ==="
TESS_FILE="n20-id1.tess"

if [ -f "$TESS_FILE" ]; then
    # Visualizes the file and prints it to img1.png using the configured 800:400 size
    neper -V "$TESS_FILE" -print img1
    echo "Success! Generated visualization saved as: img1.png"
else
    echo "Error: Expected tessellation file '$TESS_FILE' was not found."
    exit 1
fi
