#!/usr/bin/env bash
# Download and extract YSE DR1 light curves (Zenodo 7317476) into <data root>/DES dataset/YSE dataset
set -e
ROOT="${XAI_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/DES dataset/YSE dataset" && cd "$ROOT/DES dataset/YSE dataset"
curl -L -o yse_dr1_zenodo.tar.gz "https://zenodo.org/records/7317476/files/yse_dr1_zenodo.tar.gz?download=1"
tar -xzf yse_dr1_zenodo.tar.gz
echo "Extracted $(find . -name '*.snana.dat' | wc -l) light-curve files"
