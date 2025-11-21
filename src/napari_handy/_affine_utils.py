"""Affine transformation utilities for 3D manipulations."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def invert_affine(matrix):
    """Invert a 4x4 affine transformation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 affine transformation matrix

    Returns
    -------
    np.ndarray
        Inverted 4x4 affine transformation matrix
    """
    M = np.eye(4)
    M[:3, :3] = matrix[:3, :3].T
    M[:3, -1] = -matrix[:3, :3].T @ matrix[:3, -1]
    return M


def multiply_affine(matrix, x):
    """Apply affine transformation to a point.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 affine transformation matrix
    x : np.ndarray
        3D point to transform

    Returns
    -------
    np.ndarray
        Transformed 3D point
    """
    return matrix[:3, :3] @ x + matrix[:3, -1]


def build_affine_matrix(
    referential_vectors, img_shape, pos_center_hand, reference_position
):
    """Build affine transformation matrix from hand pose.

    Parameters
    ----------
    referential_vectors : np.ndarray
        3x3 rotation matrix from hand orientation
    img_shape : tuple
        Shape of the image/volume
    pos_center_hand : np.ndarray
        Current center position of hand
    reference_position : np.ndarray
        Reference position for relative movement

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix
    """
    rotation_matrix = referential_vectors.T

    # Build rotation component
    rot = np.eye(4)
    rot[:3, :3] = rotation_matrix

    # Center of volume
    origin = np.asarray(img_shape) / 2

    # Translation to origin
    T = np.eye(4)
    T[:3, -1] = origin

    # Translation from origin
    minusT = np.eye(4)
    minusT[:3, -1] = -origin

    # Combined transformation: translate to origin, rotate, translate back
    M = T @ rot @ minusT

    return M


def interpolate_affine_matrices(old_affine, new_affine, coeff):
    """Smoothly interpolate between two affine transformation matrices.

    Parameters
    ----------
    old_affine : np.ndarray
        Previous 4x4 affine matrix
    new_affine : np.ndarray
        Target 4x4 affine matrix
    coeff : float
        Interpolation coefficient (0-1)

    Returns
    -------
    np.ndarray
        Interpolated 4x4 affine matrix
    """
    interp_affine = np.eye(4)

    # Interpolate translation component linearly
    interp_affine[:3, -1] = (
        coeff * new_affine[:3, -1] + (1 - coeff) * old_affine[:3, -1]
    )

    # Interpolate rotation component using spherical linear interpolation (SLERP)
    slerp = Slerp(
        [0, 1], R.from_matrix(np.array([old_affine[:3, :3], new_affine[:3, :3]]))
    )
    interp_affine[:3, :3] = slerp(coeff).as_matrix()

    return interp_affine


def create_translation_matrix(translation):
    """Create a 4x4 translation matrix.

    Parameters
    ----------
    translation : np.ndarray
        3D translation vector

    Returns
    -------
    np.ndarray
        4x4 translation matrix
    """
    T = np.eye(4)
    T[:3, -1] = translation
    return T


def create_rotation_matrix(rotation_matrix):
    """Create a 4x4 rotation matrix from 3x3 rotation.

    Parameters
    ----------
    rotation_matrix : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    np.ndarray
        4x4 rotation matrix
    """
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = rotation_matrix
    return R_4x4
