#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m py_compile "$HERE"/*.py
g++ -O3 -std=c++17 -o /tmp/hgv_canonical_kmer_one_verify "${HERE}/canonical_kmer_one.cpp"
rm -f /tmp/hgv_canonical_kmer_one_verify
bash -n "$HERE/00_run_all_remaining_v1.sh"
echo PACKAGE_STATIC_VERIFY=PASS
