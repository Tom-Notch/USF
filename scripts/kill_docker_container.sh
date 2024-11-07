#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:04:42
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

docker exec --privileged -it "${CONTAINER_NAME}" pkill -f zsh

docker rm -f "${CONTAINER_NAME}"
