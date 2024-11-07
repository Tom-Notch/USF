#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:04:23
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

set -euo pipefail

set -a
. "$(dirname "$0")"/variables.sh
set +a

# Determine the local platform
if [[ "$(uname -m)" == "x86_64" ]]; then
	LOCAL_PLATFORM="linux/amd64"
elif [[ "$(uname -m)" == "arm64" ]]; then
	LOCAL_PLATFORM="linux/arm64"
else
	echo "Unsupported local platform: $(uname -m)"
	exit 1
fi

# Check if the builder already exists
if ! docker buildx inspect "${BUILDER}" &>/dev/null; then
	echo "Creating Docker Buildx builder '${BUILDER}'..."
	docker buildx create \
		--name "${BUILDER}" \
		--use --platform linux/amd64,linux/arm64 \
		--driver docker-container
else
	echo "Docker Buildx builder '${BUILDER}' already exists."
	docker buildx use "${BUILDER}"
fi

# Build the Docker image for only the local platform, push.sh will push multi-platform version
docker buildx bake --file "$(dirname "$0")"/../docker-compose.yml --load --set "*.platform=${LOCAL_PLATFORM}"
