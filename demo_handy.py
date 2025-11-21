"""
Demo script for napari-handy plugin.

This script demonstrates how to use the hand tracking plugin with napari.
It loads sample data and shows the hand gesture control interface.
"""

import napari
import numpy as np

from napari_handy import HandyWidget


def create_sample_data():
    """Create sample 3D data for demonstration."""
    # Create a 3D volume with some interesting features
    shape = (128, 128, 128)

    # Create a sphere in the center
    center = np.array(shape) // 2
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    distances = np.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )

    # Main volume
    volume = (distances < 40).astype(np.uint8) * 100

    # Add some noise and features
    volume += np.random.randint(0, 50, shape, dtype=np.uint8)

    # Add some bright spots
    volume[60:68, 60:68, 60:68] = 255
    volume[40:48, 40:48, 40:48] = 200

    return volume


def main():
    """Main demo function."""
    print("Starting napari-handy demo...")

    # Create napari viewer
    viewer = napari.Viewer(ndisplay=3)

    # Create and add sample data
    sample_data = create_sample_data()

    # Add the main image layer
    image_layer = viewer.add_image(
        sample_data,
        name="Sample Volume",
        opacity=0.65,
        rendering="mip",  # Maximum intensity projection
    )

    # Add a plane layer for hand control
    plane_layer = viewer.add_image(
        sample_data,
        depiction="plane",
        blending="additive",
        name="plane",
        colormap="inferno",
    )

    # Set plane to center of volume
    center = np.array(sample_data.shape) // 2
    plane_layer.plane.position = center

    # Enable 3D axes for better orientation
    viewer.axes.visible = True

    # Create and add the hand tracking widget
    hand_widget = HandyWidget(viewer)
    viewer.window.add_dock_widget(
        hand_widget, area="right", name="Hand Gesture Control"
    )

    # Run napari
    napari.run()


if __name__ == "__main__":
    main()
