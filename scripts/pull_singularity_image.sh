#!/usr/bin/env bash
#
# Created on Mon May 05 2025 13:55:27
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#

set -euo pipefail

set -a
. "$(dirname "$0")"/variables.sh
set +a

singularity build \
	--fix-perms \
	--warn-unused-build-args \
	--build-arg DOCKER_USER="${DOCKER_USER}" \
	--build-arg IMAGE_NAME="${IMAGE_NAME}" \
	--build-arg IMAGE_USER="${IMAGE_USER}" \
	--build-arg IMAGE_TAG="${IMAGE_TAG}" \
	--build-arg HOME_FOLDER="${HOME_FOLDER}" \
	--build-arg CODE_FOLDER="${CODE_FOLDER}" \
	--build-arg TIME_ZONE="${TIME_ZONE}" \
	"$(dirname "$0")/../${IMAGE_NAME}_${IMAGE_TAG}.sif" \
	"$(dirname "$0")/../singularity/default.def"
