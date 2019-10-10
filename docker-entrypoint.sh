#!/bin/sh
set -e

export FORCE_CUDA=1
cd /opt/pytorch
rm -rf vision
git config --global user.email "Lukas.Bommes@gmx.de"
git config --global user.name "LukasBommes"
git clone https://github.com/sampepose/vision.git
cd vision
pip install -v .
cd /workspace

exec "$@"
