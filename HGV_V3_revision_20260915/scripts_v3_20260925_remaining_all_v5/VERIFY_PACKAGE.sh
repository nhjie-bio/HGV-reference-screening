#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python -m py_compile ./*.py
bash -n ./00_run_all_remaining_v1.sh
g++ -std=c++17 -O2 ./canonical_kmer_one.cpp -o /tmp/hgv_canonical_kmer_one_verify
python ./PAIR_KEY_UNIT_TEST.py
python ./DASHING_MATRIX_UNIT_TEST.py
rm -f /tmp/hgv_canonical_kmer_one_verify
echo "PACKAGE_STATIC_VERIFY=PASS"
