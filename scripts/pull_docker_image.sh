#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:04:49
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

docker pull "${DOCKER_USER}"/"${IMAGE_NAME}":"${IMAGE_TAG}"
