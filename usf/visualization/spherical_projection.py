#!/usr/bin/env python3
#
# Created on Tue Nov 12 2024 17:37:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import contextlib
import os
import sys
import time
from typing import Any

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from open3d.visualization.rendering import ColorGrading

from usf.utils.spherical_image import BatchSphericalImage

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return ip is not None and ip.has_trait("kernel")  # Jupyter/VSCode notebooks
    except Exception:
        return False


def _has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    # macOS/Win usually have a display
    return True


@contextlib.contextmanager
def local_env(vars: dict[str, str], *, setdefault: bool = False):
    """
    Temporarily set environment variables, then restore previous values.
    - If setdefault=True, only set keys that aren't already defined.
    - Values are always restored exactly to their pre-block state.
    """
    _sentinel = object()
    old = {}
    try:
        for k, v in vars.items():
            old[k] = os.environ.get(k, _sentinel)
            if setdefault:
                if k not in os.environ:
                    os.environ[k] = str(v)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, prev in old.items():
            if prev is _sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


@contextlib.contextmanager
def _suppress_c_stdout_stderr():
    """
    Redirect C-level stdout (fd=1) and stderr (fd=2) into /dev/null.
    Must flush Python buffers *before* and *after* to avoid mangled output.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    # flush any pending Python-level output
    sys.stdout.flush()
    sys.stderr.flush()
    # save original fds
    orig_stdout = os.dup(1)
    orig_stderr = os.dup(2)
    # redirect both to /dev/null
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        # flush the now-silent fds (just in case)
        sys.stdout.flush()
        sys.stderr.flush()
        # restore originals
        os.dup2(orig_stdout, 1)
        os.dup2(orig_stderr, 2)
        # clean up
        os.close(devnull)
        os.close(orig_stdout)
        os.close(orig_stderr)


def visualize_spherical_image(
    batch_spherical_image: BatchSphericalImage | Any,
    point_size: float = 5.0,
    mesh_sphere_color: list = [0.7, 0.7, 0.7],
    fps: int = 1,
    loop: bool = True,
    image_size: tuple[int, int] = (1024, 1024),
    camera_center: np.ndarray | None = None,
    vertical_fov: float = 30.0,
    *,
    mode: str = "auto",
    title: str = "Spherical Image Visualization",
) -> Any | None:
    """
    Visualize a batch of SphericalImage data on a sphere using Open3D.

    Each frame in the BatchSphericalImage has:
      - vector (N, 3): Shared unit Cartesian coordinates on S^2.
      - batch_value (batch_size, N, 3): Pixel colors in [0..255] for each of batch_size frames.

    This function displays a reference sphere (radius=1) along with a point cloud
    whose points are given by vector and whose colors (normalized to [0,1]) are updated for each frame.
    The animation runs at 'fps' frames per second and loops if 'loop' is True.

    The function is written in a pure functional style: it does not modify the input instance.

    Args:
        batch_spherical_image (BatchSphericalImage | Any): Anything that can be converted to a BatchSphericalImage.
        point_size (float, optional): Size of points in the visualization. Defaults to 5.0.
        mesh_sphere_color (list): Color of the reference sphere. Defaults to [0.7, 0.7, 0.7].
        fps (int, optional): Frames per second. Defaults to 1.
        loop (bool, optional): If True, loop the animation indefinitely. Defaults to True.
        image_size (tuple[int, int], optional): rendered image size of offscreen renderer. Defaults to (1024, 1024).
        camera_center (np.ndarray | None, optional): the center of camera, will be [4, 0, 0] if not provided. Defaults to None.
        vertical_fov (float, optional): vertical field of view of offscreen renderer. Defaults to 30.0.
        mode (str, optional): "auto", "server", "native", or "offscreen". Defaults to "auto".
        title (str, optional): Title of the visualization window. Defaults to "Spherical Image Visualization".

    Returns:
        Any | None: Rendered image array for offscreen mode, None otherwise.
    """
    # --- Mode Selection ---
    chosen = mode
    if mode == "auto":
        if _in_notebook() or not _has_display():
            chosen = "server"
        elif _has_display():
            chosen = "native"
        else:
            chosen = "offscreen"

    _batch_spherical_image = batch_spherical_image

    if not isinstance(_batch_spherical_image, BatchSphericalImage):
        _batch_spherical_image = BatchSphericalImage(_batch_spherical_image)

    _batch_spherical_image = _batch_spherical_image.to_numpy().copy()
    B = len(_batch_spherical_image)

    batch_colors = _batch_spherical_image.batch_value
    num_raw_input_channels = batch_colors.shape[-1]
    assert (
        num_raw_input_channels == 1 or num_raw_input_channels == 3
    ), f"Expect # channel to be 1 or 3, but got {num_raw_input_channels}"
    if num_raw_input_channels == 1:
        batch_colors = np.repeat(batch_colors, 3, axis=-1)

    # Assumes batch_value has shape (batch_size, N, 3).
    batch_colors = batch_colors.astype(np.float32)
    if batch_colors.max() > 1.0:
        # normalize to [0, 1] if max is over 1.0
        batch_colors = batch_colors / 255.0
    batch_size = batch_colors.shape[0]

    # Create a reference sphere.
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=256)
    mesh_sphere.paint_uniform_color(mesh_sphere_color)
    mesh_sphere.compute_vertex_normals()

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultUnlit"

    # Create the point cloud and set its points once.
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(batch_colors[0])
    pcd.points = o3d.utility.Vector3dVector(_batch_spherical_image.vector)

    pcd_material = rendering.MaterialRecord()
    pcd_material.shader = "defaultUnlit"
    pcd_material.point_size = point_size
    pcd_material.sRGB_color = True

    # --- Jupyter and Server (standalone WebRTC server + browser client) ---
    if chosen == "server":
        try:
            # Enable WebRTC server if not already running, will be auto-enabled in Jupyter Notebooks.
            o3d.visualization.webrtc_server.enable_webrtc()
        except RuntimeError:
            pass

        def on_init(o3dvis: o3d.visualization.O3DVisualizer) -> None:
            """Add geometries and set up the camera when the Open3D window initializes."""
            o3dvis.add_geometry("reference sphere", mesh_sphere, mesh_material)
            o3dvis.add_geometry("spherical image", pcd, pcd_material)
            o3dvis.setup_camera(
                float(vertical_fov),
                np.array([0, 0, 0], dtype=np.float32),  # look at the center
                np.array(
                    [2.25, 0, 0], dtype=np.float32
                ),  # camera position, where your "eyes" are
                np.array([0, 0, 1], dtype=np.float32),  # up vector
            )
            return

        def on_animation_tick(
            o3dvis: o3d.visualization.O3DVisualizer,
            dt: float,
            t: float,
        ) -> o3d.visualization.O3DVisualizer.TickResult:
            """Advance to the next spherical image on each animation tick.

            Args:
                o3dvis (o3d.visualization.O3DVisualizer): The Open3D visualizer instance.
                dt (float): Time since last tick in seconds.
                t (float): Total elapsed time in seconds.

            Returns:
                o3d.visualization.O3DVisualizer.TickResult: Whether to continue.
            """
            if not hasattr(on_animation_tick, "idx"):
                on_animation_tick.idx = 0
            print(f"on_animation_tick: dt={dt}, t={t}")
            # advance on fixed timestep set by animation_time_step
            i = (int(t * max(1, fps))) % B
            if i != on_animation_tick.idx:
                on_animation_tick.idx = i

                # visible update requires remove→add
                pcd.colors = o3d.utility.Vector3dVector(batch_colors[i])
                o3dvis.update("spherical image", pcd)

                if (not loop) and i == 0 and t > 0:
                    o3dvis.is_animating = False

                return o3d.visualization.O3DVisualizer.TickResult.REDRAW
            else:
                return o3d.visualization.O3DVisualizer.TickResult.NO_CHANGE

        # this one liner import would break native rendering if imported earlier
        from open3d import web_visualizer

        web_visualizer.draw(
            [],
            title=title,
            width=image_size[0],
            height=image_size[1],
            animation_time_step=max(1e-3, 1.0 / max(1, fps)),
            on_init=on_init,
            on_animation_tick=on_animation_tick,
        )
        return None

    # --- Native Interactive Render ---
    if chosen == "native":
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh_sphere)
        vis.add_geometry(pcd)

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])  # White background
        render_option.point_size = point_size
        render_option.light_on = False

        # camera setup
        view_control = vis.get_view_control()
        view_control.set_front([1, 0, 0])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(1.0)

        time_per_frame = 1.0 / fps
        last_update_time = time.time()
        frame_idx = 1 % batch_size  # fix for when there's only 1 image

        try:
            while vis.poll_events():
                current_time = time.time()
                # Update the point cloud colors only if sufficient time has elapsed.
                if current_time - last_update_time >= time_per_frame:
                    last_update_time = current_time
                    frame_colors = batch_colors[frame_idx]  # (N, 3)
                    pcd.colors = o3d.utility.Vector3dVector(frame_colors)
                    vis.update_geometry(pcd)
                    frame_idx = (frame_idx + 1) % batch_size
                    # If we complete one pass and loop is False, exit.
                    if frame_idx == 0 and not loop:
                        break

                vis.update_renderer()
        except KeyboardInterrupt:
            pass

        vis.destroy_window()

        return None

    # --- Off-screen Render ---
    if chosen == "offscreen":
        with local_env(
            {
                "EGL_PLATFORM": "surfaceless",
                "OPEN3D_CPU_RENDERING": "true",
            },
            setdefault=True,
        ):
            with _suppress_c_stdout_stderr():
                renderer = rendering.OffscreenRenderer(*image_size)

            if camera_center is None:
                camera_center = np.array([3.9, 0, 0], dtype=np.float32)

            renderer.setup_camera(
                float(vertical_fov),
                np.array([0, 0, 0], dtype=np.float32),  # look at the center
                camera_center,  # camera position, where your "eyes" are
                np.array([0, 0, 1], dtype=np.float32),  # up vector
                -1,
                np.linalg.norm(camera_center) + 1,
            )

            scene = renderer.scene
            scene.clear_geometry()
            scene.set_background([1.0, 1.0, 1.0, 1.0])  # set a pure-white clear color
            view = scene.view
            # view.set_post_processing(False) # disable all color-grading / AO / TAA / etc.
            # view.set_ambient_occlusion(enabled = True)
            # view.set_antialiasing(enabled = True)
            # view.set_shadowing(enabled = True)
            # view.set_sample_count(8)
            view.set_color_grading(
                ColorGrading(
                    ColorGrading.Quality.ULTRA,
                    ColorGrading.ToneMapping.ACES,
                )
            )

            scene.add_geometry("spherical image", pcd, pcd_material)

            scene.add_geometry("reference sphere", mesh_sphere, mesh_material)

            img = renderer.render_to_image()
            img = np.asarray(img)

            return img

    return None
