#!/usr/bin/env bash
# Fetch only the TRL-on-Cortex recipe folder from Arctic Platform (Apache-2.0),
# pinned to a known commit, and drop the lab files next to it.
# The arctic_trl_cortex/ adapter in that folder is not shipped on PyPI.
set -euo pipefail
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="${ARCTIC_COMMIT:-a46cbadb174b7c8f79088eabd8f4d3c6065bfae2}"
DEST="${1:-$LAB/../recipe}"
SRC=recipes/rl/trl/gsm8k_cortex

tmp="$(mktemp -d)"
git clone -q --filter=blob:none --no-checkout https://github.com/Snowflake-AI-Research/Arctic-Platform "$tmp"
git -C "$tmp" sparse-checkout set "$SRC"
git -C "$tmp" checkout -q "$COMMIT"

mkdir -p "$DEST"
cp -R "$tmp/$SRC/." "$DEST/"
rm -rf "$tmp"
cp "$LAB"/{listing_task.py,run_listing_grpo_cortex.py,run_listing.sh,deploy_tuned.json} "$DEST/"
echo "Recipe + lab files in $DEST (Arctic Platform @ ${COMMIT:0:8})"
