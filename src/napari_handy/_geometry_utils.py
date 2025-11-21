"""Geometric calculations and coordinate transformations for hand tracking."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def points_coords_to_napari_coords(coords, img_shape):
    """Convert normalized hand coordinates to napari 3D coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Normalized hand coordinates (0-1 range)
    img_shape : tuple
        Shape of the image/volume

    Returns
    -------
    np.ndarray
        Coordinates in napari coordinate system
    """
    nap_coords = np.clip(coords, 0, 1)
    # Constrain X-axis to midplane for consistent positioning
    nap_coords[0] = 0.5  # Keep hand tracking constrained to midplane
    nap_coords = nap_coords * np.array(img_shape)
    return nap_coords


def compute_referential_vectors(points_data, handedness):
    """Compute reference frame vectors from hand landmarks.

    Parameters
    ----------
    points_data : np.ndarray
        Hand landmark positions
    handedness : str
        'Left' or 'Right' hand

    Returns
    -------
    np.ndarray
        3x3 rotation matrix representing hand orientation
    """
    # Key landmark vectors from wrist (point 0)
    vector_0_5 = points_data[5] - points_data[0]  # wrist to index MCP
    vector_0_9 = points_data[9] - points_data[0]  # wrist to middle MCP
    vector_0_13 = points_data[13] - points_data[0]  # wrist to ring MCP
    vector_0_17 = points_data[17] - points_data[0]  # wrist to pinky MCP

    # Compute normal vector (z-axis) using cross products
    vector_z = np.mean(
        [
            np.cross(vector_0_9, vector_0_5),
            np.cross(vector_0_13, vector_0_9),
            np.cross(vector_0_17, vector_0_13),
        ],
        axis=0,
    )

    # Flip for left hand to maintain consistent orientation
    if handedness == "Left":
        vector_z *= -1

    # Y-axis points toward fingers
    vector_y = -np.mean([vector_0_5, vector_0_9, vector_0_17, vector_0_13], axis=0)

    # X-axis completes right-handed coordinate system
    vector_x = np.cross(vector_z, vector_y)

    # Normalize and create rotation matrix
    referential_vectors = np.array([vector_z, vector_y, vector_x])
    referential_vectors = referential_vectors / np.linalg.norm(
        referential_vectors, axis=1
    ).reshape(-1, 1)

    referential_vectors = R.from_matrix(referential_vectors).as_matrix()
    return referential_vectors


def determine_if_holding(points_data):
    """Determine if hand is in holding/grasping gesture.

    Parameters
    ----------
    points_data : np.ndarray
        Hand landmark positions

    Returns
    -------
    bool
        True if hand is holding/grasping
    """
    # Distance between thumb tip and index tip
    distance_marker = np.linalg.norm(points_data[4] - points_data[8])
    # Reference distance (thumb tip to thumb joint)
    distance_reference = np.linalg.norm(points_data[3] - points_data[4])

    return distance_marker < distance_reference * 1.2


def compute_hand_center(points_data):
    """Compute center position of hand from key landmarks.

    Parameters
    ----------
    points_data : np.ndarray
        Hand landmark positions

    Returns
    -------
    np.ndarray
        Center position of hand
    """
    return np.mean(
        [
            points_data[0],  # wrist
            points_data[5],  # index MCP
            points_data[17],  # pinky MCP
            points_data[9],  # middle MCP
            points_data[13],  # ring MCP
        ],
        axis=0,
    )
