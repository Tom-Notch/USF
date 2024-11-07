#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:04:56
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

# docker push "${DOCKER_USER}"/"${IMAGE_NAME}":"${IMAGE_TAG}"

docker buildx bake --file "$(dirname "$0")"/../docker-compose.yml --push
