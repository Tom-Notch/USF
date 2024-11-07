#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 18:04:36
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

. "$(dirname "$0")"/variables.sh

echo "Removing all containers"

for id in $(docker ps -aq); do
	echo "Removing container $id"
	docker rm -f "$id"
done

echo "Done"
