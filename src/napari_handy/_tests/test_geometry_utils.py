"""Tests for geometry utilities."""

import numpy as np

from napari_handy._geometry_utils import (
    compute_hand_center,
    compute_referential_vectors,
    determine_if_holding,
    points_coords_to_napari_coords,
)


def test_points_coords_to_napari_coords():
    """Test coordinate conversion to napari format."""
    coords = np.array([0.2, 0.5, 0.8])
    img_shape = (100, 200, 300)

    result = points_coords_to_napari_coords(coords, img_shape)

    # Check that coordinates are properly scaled
    assert result[0] == 50.0  # Should be constrained to midplane
    assert result[1] == 100.0  # 0.5 * 200
    assert result[2] == 240.0  # 0.8 * 300


def test_points_coords_clipping():
    """Test coordinate clipping behavior."""
    coords = np.array([-0.5, 1.5, 0.5])  # Out of bounds
    img_shape = (100, 100, 100)

    result = points_coords_to_napari_coords(coords, img_shape)

    # Check clipping
    assert result[0] == 50.0  # Constrained to midplane
    assert result[1] == 100.0  # Clipped to max
    assert result[2] == 50.0


def test_compute_hand_center():
    """Test hand center computation."""
    # Create mock hand landmarks (21 points, 3D)
    points_data = np.random.rand(21, 3)

    # Set specific landmarks for known positions
    points_data[0] = [0.1, 0.1, 0.1]  # wrist
    points_data[5] = [0.2, 0.2, 0.2]  # index MCP
    points_data[9] = [0.3, 0.3, 0.3]  # middle MCP
    points_data[13] = [0.4, 0.4, 0.4]  # ring MCP
    points_data[17] = [0.5, 0.5, 0.5]  # pinky MCP

    center = compute_hand_center(points_data)

    # Should be mean of the 5 key points
    expected = np.array([0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(center, expected)


def test_determine_if_holding():
    """Test holding gesture detection."""
    # Create mock hand landmarks
    points_data = np.zeros((21, 3))

    # Test holding case (thumb and index close)
    points_data[4] = [0.1, 0.1, 0.1]  # thumb tip
    points_data[8] = [0.1, 0.1, 0.1]  # index tip (same position)
    points_data[3] = [0.0, 0.0, 0.0]  # thumb joint

    is_holding = determine_if_holding(points_data)
    assert is_holding == True

    # Test not holding case (thumb and index far apart)
    points_data[8] = [10.0, 10.0, 10.0]  # index tip far away

    is_holding = determine_if_holding(points_data)
    assert is_holding == False


def test_compute_referential_vectors():
    """Test reference frame computation."""
    # Create mock hand landmarks with known geometry
    points_data = np.zeros((21, 3))

    # Set up realistic hand geometry that avoids zero vectors
    # Wrist at origin, fingers extending outward and slightly forward
    points_data[0] = [0.0, 0.0, 0.0]  # wrist at origin
    points_data[5] = [0.05, 0.08, 0.01]  # index MCP 
    points_data[9] = [0.02, 0.10, 0.01]  # middle MCP
    points_data[13] = [-0.02, 0.09, 0.01]  # ring MCP
    points_data[17] = [-0.05, 0.07, 0.01]  # pinky MCP

    # Test for right hand
    vectors = compute_referential_vectors(points_data, "Right")

    # Should return a 3x3 rotation matrix
    assert vectors.shape == (3, 3)
    
    # Check that we don't have NaN values
    assert not np.any(np.isnan(vectors))
    
    # Check that it's a proper rotation matrix (orthogonal with det=1)
    should_be_identity = np.dot(vectors, vectors.T)
    assert np.allclose(should_be_identity, np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(vectors), 1.0, atol=1e-10)

    # Test for left hand
    vectors_left = compute_referential_vectors(points_data, "Left")
    assert vectors_left.shape == (3, 3)
    assert not np.any(np.isnan(vectors_left))
    
    # Check that it's also a proper rotation matrix
    should_be_identity_left = np.dot(vectors_left, vectors_left.T)
    assert np.allclose(should_be_identity_left, np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(vectors_left), 1.0, atol=1e-10)

    # Left and right should differ due to handedness flip
    assert not np.allclose(vectors, vectors_left)
