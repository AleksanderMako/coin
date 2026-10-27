#!/bin/bash
# Prepare a fresh RunPod pod: clone the branch, install, make the output folders.
# Usage on the pod: bash bootstrap.sh <branch>
set -e
BRANCH=${1:-feature/sharpness-and-learned-start}
cd /workspace
if [ ! -d coin ]; then
  git clone -b "$BRANCH" https://github.com/AleksanderMako/coin.git
fi
cd coin
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git reset -q --hard "origin/$BRANCH"
pip install --break-system-packages -q imageio tqdm matplotlib
mkdir -p runs logs
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
