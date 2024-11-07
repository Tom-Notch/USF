#!/usr/bin/env bash
#
# Created on Fri Oct 10 2025 23:36:49
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#

set -euo pipefail
. "$(dirname "$0")"/variables.sh

singularity instance stop "${CONTAINER_NAME}"
