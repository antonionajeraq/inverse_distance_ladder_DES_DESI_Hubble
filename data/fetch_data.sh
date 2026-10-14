#!/usr/bin/env bash
# Download the four input files from their upstream repositories, at the exact commits the
# paper used, and verify each against its SHA-256.
#
# The files are not redistributed here: neither upstream repository carries a licence that
# grants it.  Pinning the commit and checking the checksum makes the download exactly as
# reproducible as shipping the files would be.  dovekie_cosmography also re-checks every
# file at read time.
#
# Usage:  bash data/fetch_data.sh          # from anywhere; skips files already present
#         bash data/fetch_data.sh --force  # download again even if present
set -euo pipefail
cd "$(dirname "$0")"
FORCE="${1:-}"

# DES-Dovekie (Popovic et al. 2026), des-science/DES-SN5YR, 4_DISTANCES_COVMAT/
DES="https://raw.githubusercontent.com/des-science/DES-SN5YR/442c24841b55c3d629ed7177c09763dc237ba8e1/4_DISTANCES_COVMAT"
# Moresco et al. (2020) chronometer covariance, mmoresco/CCcovariance, data/
CC="https://gitlab.com/mmoresco/CCcovariance/-/raw/881413330a7f1e1e5203607d6964db49b4c6c461/data"

fetch () {         # fetch <url> <destination> <expected sha256>
    local url="$1" dest="$2" want="$3"
    if [ -f "$dest" ] && [ "$FORCE" != "--force" ]; then
        if [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
            echo "  $dest  present, OK"; return
        fi
        echo "  $dest  present but its checksum is wrong; downloading again"
    fi
    mkdir -p "$(dirname "$dest")"
    local tmp="$dest.part"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$tmp"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$tmp"
    else
        echo "need curl or wget" >&2; exit 1
    fi
    local got
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$tmp"
        echo "  *** CHECKSUM MISMATCH for $dest" >&2
        echo "      got      $got" >&2
        echo "      expected $want" >&2
        echo "      This is not the file the published numbers were computed from." >&2
        exit 1
    fi
    mv "$tmp" "$dest"
    echo "  $dest  downloaded, OK"
}

echo "Fetching the inputs (see data/PROVENANCE.md)"
fetch "$DES/DES-Dovekie_HD.csv" DES-Dovekie_HD.csv \
      2f57019d783eaa976df80a41b0054171a2d994ee9808d715ce850c2df5720aaf
fetch "$DES/STAT%2BSYS.npz" STAT+SYS.npz \
      ffd3124b32148b1372bd95fda9299269f0352a9f8eee02d416c610e38495463b
fetch "$CC/HzTable_MM_BC03.dat" CC_mm20/HzTable_MM_BC03.dat \
      32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98
fetch "$CC/data_MM20.dat" CC_mm20/data_MM20.dat \
      577ac2f346e346fe7cf94daa7b7000c05d04ebc8a029cda31e0d8643b956a485
echo "All inputs present and verified."
