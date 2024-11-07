#!/usr/bin/env bash
#
# Created on Wed Sep 04 2024 14:53:41
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#

set -e # Exit immediately if a command exits with a non-zero status

# Prompt for the sudo password once and store it in a variable
echo "Please enter your sudo password:"
read -rs SUDO_PASSWORD # Use -r to avoid mangling backslashes, -s for silent input

# Keep sudo alive during the script execution
echo "$SUDO_PASSWORD" | sudo -v -S

# Update package lists
echo "$SUDO_PASSWORD" | sudo -S apt update

# Install dependencies
echo "Installing clang-format clang-tidy python3 python3-pip"
echo "$SUDO_PASSWORD" | sudo -S apt install -y clang-format clang-tidy libpython3-dev python3-pip

# Install/upgrade pre-commit
echo "Installing/upgrading pre-commit"
uv pip install --upgrade pre-commit

# Detect the user's default shell
DEFAULT_SHELL=$(basename "$SHELL")

# Determine the shell configuration file
case "$DEFAULT_SHELL" in
bash)
	SHELL_RC="${HOME}/.bashrc"
	;;
zsh)
	SHELL_RC="${HOME}/.zshrc"
	;;
fish)
	SHELL_RC="${HOME}/.config/fish/config.fish"
	;;
ksh)
	SHELL_RC="${HOME}/.kshrc"
	;;
*)
	echo "Unsupported shell: $DEFAULT_SHELL"
	exit 1
	;;
esac

# Add pre-commit executable to PATH if not already present
if ! grep -q 'export PATH=~/.local/bin:${PATH}' "$SHELL_RC"; then
	echo "Adding pre-commit (actually python3-pip packages') executable to path in $SHELL_RC"
	echo 'export PATH=~/.local/bin:${PATH}' >>"$SHELL_RC"
	# Source the .zshrc file using zsh
	if [ "$DEFAULT_SHELL" = "zsh" ]; then
		zsh -c "source $SHELL_RC"
	else
		# shellcheck source=/dev/null
		. "$SHELL_RC"
	fi
else
	echo "PATH already updated in $SHELL_RC"
fi

# Perform pre-installation of pre-commit hooks and dry run on all files
echo "Performing pre-installation of pre-commit hooks and dry run on all files"
pre-commit run --all-files --hook-stage manual

# Set up pre-commit locally in this repository
echo "Setting up pre-commit locally in this repository"
pre-commit install

echo "Done!"
