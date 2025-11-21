"""Tests for hand tracker module."""

import numpy as np

from napari_handy._hand_tracker import HandState, HandTracker


def test_hand_tracker_initialization():
    """Test HandTracker initialization."""
    tracker = HandTracker()
    assert tracker.min_detection_confidence == 0.8
    assert tracker.min_tracking_confidence == 0.5
    assert len(tracker.handedness_buffer) == 0


def test_hand_state_initialization():
    """Test HandState initialization."""
    state = HandState()
    assert state.is_detected is False
    assert state.landmarks is None
    assert state.center is None
    assert state.is_holding is False


def test_hand_state_update():
    """Test HandState update with mock data."""
    state = HandState()

    # Mock hand data
    landmarks = np.random.rand(21, 3)
    hand_data = {
        "landmarks": landmarks,
        "center": np.array([0.5, 0.5, 0.5]),
        "orientation": np.eye(3),
        "is_holding": True,
        "handedness": "Right",
    }

    state.update(hand_data)

    assert state.is_detected is True
    assert np.array_equal(state.landmarks, landmarks)
    assert state.handedness == "Right"


def test_hand_state_reset():
    """Test HandState reset functionality."""
    state = HandState()

    # Set some state
    state.is_detected = True
    state.landmarks = np.random.rand(21, 3)

    # Reset
    state.reset()

    assert state.is_detected is False
    assert state.landmarks is None


def test_hand_state_get_state():
    """Test getting hand state as dictionary."""
    state = HandState()
    state.is_detected = True
    state.landmarks = np.random.rand(21, 3)
    state.center = np.array([0.5, 0.5, 0.5])
    state.handedness = "Left"

    state_dict = state.get_state()

    assert isinstance(state_dict, dict)
    assert "landmarks" in state_dict
    assert "center" in state_dict
    assert "handedness" in state_dict
