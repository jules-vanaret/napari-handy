"""Camera preview widget for napari-handy."""

import numpy as np
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from skimage.draw import disk


class CameraPreviewWidget(QWidget):
    """Widget to display camera feed in napari."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Timer for updating preview
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        
        # Current frame data
        self.current_frame = None
        self.current_hand_landmarks = None  # Store hand landmarks for overlay
        self.preview_enabled = False
        
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout()
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: #666666;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera Preview\n(Start tracking to view)")
        self.video_label.setScaledContents(True)
        
        layout.addWidget(self.video_label)
        self.setLayout(layout)
    
    def start_preview(self):
        """Start the camera preview."""
        self.preview_enabled = True
        self.update_timer.start(100)  # ~10 FPS update rate
    
    def stop_preview(self):
        """Stop the camera preview."""
        self.preview_enabled = False
        self.update_timer.stop()
        self.video_label.setText("Camera Preview\n(Start tracking to view)")
        self.current_frame = None
        self.current_hand_landmarks = None
    
    def update_frame(self, frame: np.ndarray, hand_landmarks=None):
        """Update the current frame and hand landmarks.
        
        Parameters
        ----------
        frame : np.ndarray
            RGB frame from camera
        hand_landmarks : mediapipe hand landmarks, optional
            Hand landmarks from the hand with highest confidence
        """
        if self.preview_enabled:
            self.current_frame = frame.copy()
            self.current_hand_landmarks = hand_landmarks
    
    def _update_display(self):
        """Update the display with current frame and hand overlay."""
        if not self.preview_enabled or self.current_frame is None:
            return
        
        # Apply hand landmarks overlay if available
        frame = self.current_frame
        if self.current_hand_landmarks is not None:
            frame = self.draw_hand_overlay(frame, self.current_hand_landmarks)
        
        height, width, channels = frame.shape
        
        # Convert to QImage
        bytes_per_line = channels * width
        q_image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Scale to fit widget while maintaining aspect ratio
        label_size = self.video_label.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def draw_hand_overlay(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """Draw hand landmarks on frame using the hand with highest confidence.
        
        Parameters
        ----------
        frame : np.ndarray
            RGB frame
        hand_landmarks : mediapipe hand landmarks
            Hand landmarks to draw (from the hand with highest confidence)
            
        Returns
        -------
        np.ndarray
            Frame with hand landmarks overlay
        """
        if hand_landmarks is None:
            return frame.copy()
        
        # Create a copy of the frame to draw on
        frame_with_landmarks = frame.copy()
        height, width = frame.shape[:2]
        
        # Define colors for different landmark types (RGB)
        landmark_colors = {
            'wrist': (255, 255, 255),        # White
            'thumb': (255, 0, 0),            # Red  
            'index': (0, 255, 0),            # Green
            'middle': (0, 0, 255),           # Blue
            'ring': (255, 255, 0),           # Yellow
            'pinky': (255, 0, 255),          # Magenta
        }
        
        # MediaPipe hand landmark indices grouped by finger
        landmark_groups = {
            'wrist': [0],
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # Draw landmarks
        for finger, indices in landmark_groups.items():
            color = landmark_colors[finger]
            
            for idx in indices:
                if idx < len(hand_landmarks.landmark):
                    landmark = hand_landmarks.landmark[idx]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    
                    # Ensure coordinates are within frame bounds
                    if 0 <= x < width and 0 <= y < height:
                        # Draw landmark as a filled circle using skimage.draw
                        radius = 10 if finger == 'wrist' else 8
                        
                        try:
                            # Get circle coordinates
                            rr, cc = disk((y, x), radius, shape=(height, width))
                            
                            # Draw the landmark
                            frame_with_landmarks[rr, cc] = color
                            
                        except (IndexError, ValueError):
                            # Skip if coordinates are invalid
                            continue
        
        return frame_with_landmarks