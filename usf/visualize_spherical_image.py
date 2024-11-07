#!/usr/bin/env python3
#
# Created on Mon Feb 17 2025 13:05:07
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
"""Interactively visualize a spherical image (``.npz`` or ``.bin``) using Open3D.

CLI entry point installed as ``visualize_spherical_image``::

    visualize_spherical_image -p path/to/image.npz [-f fps] [-s point_size]
"""

import argparse

from usf import BASE_DIR
from usf.utils.files import parse_path
from usf.visualization.spherical_projection import visualize_spherical_image


def main() -> None:
    """Parse CLI args and launch the interactive Open3D spherical image viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, help="Path to the configuration file.", required=True
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=float,
        help="visualization frame per second to switch images",
        default=1.0,
    )
    parser.add_argument(
        "--point_size",
        "-s",
        type=float,
        help="display size of each point on the sphere",
        default=5.0,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["auto", "server", "native"],
        default="auto",
        help="Visualization mode: 'auto' for auto-determine, 'server' for web server, 'native' for native machine visualization",
    )
    args = parser.parse_args()
    spherical_image_path = parse_path(args.path, BASE_DIR)

    print(f"Reading spherical image: {spherical_image_path}")

    visualize_spherical_image(
        spherical_image_path,
        point_size=args.point_size,
        fps=args.fps,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
