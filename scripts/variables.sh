#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:05:17
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

set -euo pipefail

resolve_host_path() {
	local p="$1"
	[ -n "$p" ] || {
		echo ""
		return 0
	}

	if command -v realpath >/dev/null 2>&1; then
		realpath -m -- "$p" 2>/dev/null || echo "$p"
	elif readlink -f / >/dev/null 2>&1; then
		readlink -f -- "$p" 2>/dev/null || echo "$p"
	else
		python3 - "$p" <<'PY' 2>/dev/null || echo "$p"
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
	fi
}

set -a
set +u
. "$(dirname "$0")"/../.env
set -u
set +a

# export any additional derived variables here if needed, e.g. resolved absolute paths
BASE_FOLDER="$(resolve_host_path "$(dirname "$0")"/..)"
export BASE_FOLDER

MNIST_DIR="$(resolve_host_path "${MNIST_DIR}")"
export MNIST_DIR
PANDORA_DIR="$(resolve_host_path "${PANDORA_DIR}")"
export PANDORA_DIR
STANFORD2D3DS_DIR="$(resolve_host_path "${STANFORD2D3DS_DIR}")"
export STANFORD2D3DS_DIR
