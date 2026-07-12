#!/bin/sh
set -e

# /app/data is a volume mount (often an empty host bind mount on Coolify),
# which shadows anything baked into the image at that path. Seed it from
# the image on first run without clobbering an already-trained volume.
mkdir -p /app/data/raw
[ -f /app/data/raw/stroke.csv ] || cp /app/seed/raw/stroke.csv /app/data/raw/stroke.csv
[ -f /app/data/best_params.json ] || cp /app/seed/best_params.json /app/data/best_params.json

exec "$@"
