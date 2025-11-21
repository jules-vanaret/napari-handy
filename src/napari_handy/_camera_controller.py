"""Camera controller for OpenCV camera handling with lazy imports."""

from time import time

from napari.qt.threading import thread_worker


class CameraController:
    """Manages camera access and frame capture with lazy OpenCV import."""

    def __init__(self, camera_id=0):
        """Initialize camera controller.

        Parameters
        ----------
        camera_id : int
            Camera device ID (default: 0 for default camera)
        """
        self.camera_id = camera_id
        self._cap = None
        self._cv2 = None

    def _lazy_import_cv2(self):
        """Lazy import of OpenCV to avoid napari conflicts."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def initialize_camera(self):
        """Initialize camera capture.

        Returns
        -------
        bool
            True if camera initialized successfully
        """
        cv2 = self._lazy_import_cv2()

        try:
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                return False
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False

    def capture_frame(self):
        """Capture a single frame from camera.

        Returns
        -------
        np.ndarray or None
            RGB frame array, None if capture failed
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        cv2 = self._lazy_import_cv2()

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Convert BGR to RGB and flip horizontally for mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 1)

        # Set flag for MediaPipe processing
        rgb_frame.flags.writeable = False

        return rgb_frame

    def is_opened(self):
        """Check if camera is opened and ready.

        Returns
        -------
        bool
            True if camera is ready
        """
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._cv2 is not None:
            self._cv2.destroyAllWindows()

    def __enter__(self):
        """Context manager entry."""
        if not self.initialize_camera():
            raise RuntimeError("Failed to initialize camera")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


@thread_worker
def camera_worker(hand_tracker, callback, framerate=20, camera_id=0):
    """Thread worker for camera capture and hand detection.

    Parameters
    ----------
    hand_tracker : HandTracker
        Hand tracker instance
    callback : callable
        Function to call with detection results
    framerate : int
        Target framerate for processing
    camera_id : int
        Camera device ID
    """
    # Lazy import cv2 inside the worker thread
    import cv2

    time_start = time()
    frame_interval = 1.0 / framerate

    # Initialize camera in the worker thread
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Camera worker error: Failed to initialize camera {camera_id}")
        return

    try:
        while cap.isOpened():
            # Receive control messages
            message = yield

            if message == "stop":
                break
            elif isinstance(message, dict):
                # Update parameters from message
                if "framerate" in message:
                    framerate = message["framerate"]
                    frame_interval = 1.0 / framerate

            # Process at target framerate
            current_time = time()
            if current_time - time_start >= frame_interval:
                ret, frame = cap.read()

                if ret and frame is not None:
                    # Convert BGR to RGB and flip horizontally for mirror effect
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = cv2.flip(rgb_frame, 1)

                    # Set flag for MediaPipe processing
                    rgb_frame.flags.writeable = False

                    # Process frame with hand tracker
                    results = hand_tracker.process_frame(rgb_frame)

                    # Send results to callback
                    if callback and results:
                        callback(results)

                time_start = current_time

    except Exception as e:
        print(f"Camera worker error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
