#!/bin/sh
set -e
# Create dummy DINOv2 head if missing (so app can start before training)
if [ ! -f weights/body_part_dinov2_head.pt ]; then
    echo "Creating dummy weights (run training to replace with MURA-trained head)..."
    python scripts/create_dummy_weights.py
fi
exec "$@"
