"""Hand tracking logic using MediaPipe with lazy imports."""

from collections import deque

import numpy as np

from ._geometry_utils import (
    compute_hand_center,
    compute_referential_vectors,
    determine_if_holding,
)


class HandTracker:
    """Encapsulates MediaPipe hand tracking with state management."""

    def __init__(
        self,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        handedness_buffer_size=5,
    ):
        """Initialize hand tracker.

        Parameters
        ----------
        min_detection_confidence : float
            Minimum confidence for hand detection
        min_tracking_confidence : float
            Minimum confidence for hand tracking
        handedness_buffer_size : int
            Size of buffer for handedness smoothing
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Lazy import attributes
        self._mp = None
        self._mp_hands = None
        self._mp_drawing = None
        self._hands_model = None

        # State tracking
        self.handedness_buffer = deque(maxlen=handedness_buffer_size)
        self.left_hand_state = HandState()
        self.right_hand_state = HandState()

        # Detection history for smoothing
        self.detection_history = deque(maxlen=3)

    def _lazy_import_mediapipe(self):
        """Lazy import of MediaPipe to avoid conflicts."""
        if self._mp is None:
            try:
                import mediapipe as mp

                self._mp = mp
                self._mp_drawing = mp.solutions.drawing_utils
                self._mp_hands = mp.solutions.hands
            except ImportError as e:
                raise ImportError(
                    f"MediaPipe not found. Please install with: pip install mediapipe>=0.10.0\nError: {e}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import MediaPipe. This might be a protobuf version issue. Try: pip install 'protobuf>=3.20.0,<4.0.0'\nError: {e}"
                )
        return self._mp, self._mp_hands, self._mp_drawing

    def initialize(self):
        """Initialize MediaPipe hands model.

        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            mp, mp_hands, _ = self._lazy_import_mediapipe()

            self._hands_model = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            return False
        except Exception as e:
            print(f"Failed to initialize MediaPipe: {e}")
            print(
                "Try running: pip install 'protobuf>=3.20.0,<4.0.0' mediapipe>=0.10.0"
            )
            return False

    def process_frame(self, rgb_frame):
        """Process a single frame for hand detection.

        Parameters
        ----------
        rgb_frame : np.ndarray
            RGB frame from camera

        Returns
        -------
        dict or None
            Detection results with hand data
        """
        if self._hands_model is None:
            if not self.initialize():
                return None

        try:
            # Process frame with MediaPipe
            results = self._hands_model.process(rgb_frame)

            if results.multi_hand_world_landmarks is None:
                # No hands detected
                self.left_hand_state.reset()
                self.right_hand_state.reset()
                return {
                    "left_hand": None,
                    "right_hand": None,
                    "left_hand_raw": None,
                    "right_hand_raw": None,
                    "frame": rgb_frame,
                    "frame_shape": rgb_frame.shape,
                }

            # Extract hand data
            hand_data = self._extract_hand_data(results)

            # Update hand states
            self._update_hand_states(hand_data)
            
            # Get handedness list for raw data mapping
            handedness_list = [
                hand.classification[0].label 
                for hand in results.multi_handedness
            ]

            return {
                "left_hand": (
                    self.left_hand_state.get_state()
                    if self.left_hand_state.is_detected
                    else None
                ),
                "right_hand": (
                    self.right_hand_state.get_state()
                    if self.right_hand_state.is_detected
                    else None
                ),
                "left_hand_raw": (
                    results.multi_hand_landmarks[handedness_list.index('Left')] 
                    if 'Left' in handedness_list else None
                ),
                "right_hand_raw": (
                    results.multi_hand_landmarks[handedness_list.index('Right')] 
                    if 'Right' in handedness_list else None
                ),
                "frame": rgb_frame,
                "frame_shape": rgb_frame.shape,
            }

        except Exception as e:
            print(f"Hand tracking error: {e}")
            return None

    def _extract_hand_data(self, results):
        """Extract hand landmark data from MediaPipe results.

        Parameters
        ----------
        results : mediapipe.framework.formats.detection_pb2.DetectionList
            MediaPipe detection results

        Returns
        -------
        dict
            Extracted hand data by handedness
        """
        hand_data = {}

        if results.multi_handedness and results.multi_hand_landmarks:
            handedness_list = [
                hand.classification[0].label for hand in results.multi_handedness
            ]

            for handedness, hand_landmarks in zip(
                handedness_list, results.multi_hand_landmarks, strict=False
            ):
                # Extract 3D landmark positions (z, y, x format for napari)
                points_data = np.array(
                    [
                        [landmark.z, landmark.y, landmark.x]
                        for landmark in hand_landmarks.landmark
                    ]
                )
                # print(hand_landmarks.landmark[5].z, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].x)
                # Compute derived properties
                center_position = compute_hand_center(points_data)
                # print(center_position)
                referential_vectors = compute_referential_vectors(
                    points_data, handedness
                )
                is_holding = determine_if_holding(points_data)

                hand_data[handedness] = {
                    "landmarks": points_data,
                    "center": center_position,
                    "orientation": referential_vectors,
                    "is_holding": is_holding,
                    "handedness": handedness,
                }

        return hand_data

    def _update_hand_states(self, hand_data):
        """Update internal hand states with new detection data.

        Parameters
        ----------
        hand_data : dict
            Hand data by handedness
        """
        # Reset states
        self.left_hand_state.reset()
        self.right_hand_state.reset()

        # Update with detected hands
        if "Left" in hand_data:
            self.left_hand_state.update(hand_data["Left"])

        if "Right" in hand_data:
            self.right_hand_state.update(hand_data["Right"])

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self._hands_model is not None:
            self._hands_model.close()
            self._hands_model = None


class HandState:
    """Tracks state of a single hand over time."""

    def __init__(self):
        """Initialize hand state."""
        self.is_detected = False
        self.landmarks = None
        self.center = None
        self.orientation = None
        self.is_holding = False
        self.was_holding = False
        self.holding_start_position = None
        self.handedness = None

        # State history for smoothing
        self.position_history = deque(maxlen=3)
        self.holding_history = deque(maxlen=3)

    def update(self, hand_data):
        """Update hand state with new detection.

        Parameters
        ----------
        hand_data : dict
            Hand detection data
        """
        self.is_detected = True
        self.landmarks = hand_data["landmarks"]
        self.center = hand_data["center"]
        self.orientation = hand_data["orientation"]
        self.handedness = hand_data["handedness"]

        # Update holding state
        current_holding = hand_data["is_holding"]
        self.holding_history.append(current_holding)

        # Smooth holding detection
        if len(self.holding_history) >= 2:
            holding_votes = sum(self.holding_history)
            smoothed_holding = holding_votes >= len(self.holding_history) // 2
        else:
            smoothed_holding = current_holding

        # Detect holding state changes
        if smoothed_holding and not self.was_holding:
            # Started holding
            self.holding_start_position = self.center.copy()
        elif not smoothed_holding and self.was_holding:
            # Stopped holding
            self.holding_start_position = None

        self.was_holding = self.is_holding
        self.is_holding = smoothed_holding

        # Update position history
        self.position_history.append(self.center.copy())

    def reset(self):
        """Reset hand state when not detected."""
        self.is_detected = False
        self.landmarks = None
        self.center = None
        self.orientation = None
        # Keep holding state briefly to avoid flickering

    def get_state(self):
        """Get current hand state as dict.

        Returns
        -------
        dict
            Current hand state
        """
        return {
            "landmarks": self.landmarks,
            "center": self.center,
            "orientation": self.orientation,
            "is_holding": self.is_holding,
            "was_holding": self.was_holding,
            "holding_start_position": self.holding_start_position,
            "handedness": self.handedness,
        }

    def get_smoothed_position(self):
        """Get position smoothed over recent history.

        Returns
        -------
        np.ndarray or None
            Smoothed center position
        """
        if not self.position_history:
            return self.center

        return np.mean(list(self.position_history), axis=0)
