"""Hand tracking widget for napari using gesture control."""

from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory, magicgui
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QTextEdit,
)

from ._affine_utils import (
    invert_affine,
    multiply_affine,
)
from ._camera_controller import camera_worker
from ._camera_preview import CameraPreviewWidget
from ._depth_calibrator import DepthCalibrator
from ._geometry_utils import points_coords_to_napari_coords
from ._hand_tracker import HandTracker
from magicgui.widgets import CheckBox, Container, create_widget
from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari


class HandyWidget(QWidget):
    """Main hand tracking widget for napari."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Core components
        self.hand_tracker = HandTracker()
        self.depth_calibrator = DepthCalibrator()
        self.camera_preview = CameraPreviewWidget()
        self.worker = None

        # State
        self.is_tracking = False
        self.tracking_layers = {}
        self.old_affine_matrix = None
        self.new_affine_matrix = None
        self.marking = False
        self.calibration_mode = False

        # Parameters
        self.framerate = 20
        self.interp_coeff = 0.6

        self._setup_ui()
        self._setup_layers()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()

        # Add informational tip at the top
        tip_label = QLabel("""
<b>Tip:</b> Hand recognition works best with good lighting and a uniform background (plain wall, solid color).
        """)
        tip_label.setWordWrap(True)

        layout.addWidget(tip_label)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Create tabs
        self.tracking_tab = self._create_tracking_tab()
        self.calibration_tab = self._create_calibration_tab()
        
        # Add tabs
        self.tab_widget.addTab(self.tracking_tab, "Hand Tracking")
        self.tab_widget.addTab(self.calibration_tab, "Depth Calibration")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _create_tracking_tab(self):
        """Create the main tracking tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Hand tracking: Stopped")
        layout.addWidget(self.status_label)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self._start_tracking)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.clicked.connect(self._stop_tracking)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        layout.addLayout(controls_layout)

        # Parameter controls
        self.params_widget = self._create_parameter_controls()
        layout.addWidget(self.params_widget.native)  # Use .native to get the QWidget
        
        layout.addStretch(1)
        tab.setLayout(layout)

        return tab
    
    def _create_calibration_tab(self):
        """Create the depth calibration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("""
<b>Depth Calibration Instructions:</b><br>
1. Start tracking first<br>
2. Hold one hand <b>close</b> to the camera (palm open) and click "Capture Near"<br>
3. Hold the same hand <b>far</b> from the camera (palm open) and click "Capture Far"<br>
4. Click "Finalize Calibration" to complete setup<br>
<br>
The camera preview shows what the system sees.
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Camera preview
        layout.addWidget(self.camera_preview)
        
        # Calibration status
        self.calib_status_label = QLabel("Status: Not calibrated")
        layout.addWidget(self.calib_status_label)
        
        # Calibration buttons
        calib_buttons_layout = QHBoxLayout()
        
        self.capture_near_btn = QPushButton("Capture Near")
        self.capture_near_btn.clicked.connect(self._capture_near)
        self.capture_near_btn.setEnabled(False)
        calib_buttons_layout.addWidget(self.capture_near_btn)
        
        self.capture_far_btn = QPushButton("Capture Far")
        self.capture_far_btn.clicked.connect(self._capture_far)
        self.capture_far_btn.setEnabled(False)
        calib_buttons_layout.addWidget(self.capture_far_btn)
        
        layout.addLayout(calib_buttons_layout)
        
        self.finalize_btn = QPushButton("Finalize Calibration")
        self.finalize_btn.clicked.connect(self._finalize_calibration)
        self.finalize_btn.setEnabled(False)
        layout.addWidget(self.finalize_btn)
        
        self.reset_calib_btn = QPushButton("Reset Calibration")
        self.reset_calib_btn.clicked.connect(self._reset_calibration)
        layout.addWidget(self.reset_calib_btn)
        
        # Calibration info display
        self.calib_info = QTextEdit()
        self.calib_info.setMaximumHeight(100)
        self.calib_info.setReadOnly(True)
        layout.addWidget(self.calib_info)

        layout.addStretch(1)
        tab.setLayout(layout)
        
        return tab

    def _create_parameter_controls(self):
        """Create parameter control widget."""

        @magicgui(
            auto_call=True,
            framerate={"widget_type": "Slider", "min": 1, "max": 30, "step": 1},
            interp_coeff={
                "widget_type": "FloatSlider",
                "min": 0.01,
                "max": 0.99,
                "step": 0.01,
            },
            call_button=False,
        )
        def parameter_controls(
            framerate: int = 20,
            interp_coeff: float = 0.6,
        ):
            self.framerate = framerate
            self.interp_coeff = interp_coeff

            # Update worker if running
            if self.worker is not None:
                self.worker.send({"framerate": framerate})

        return parameter_controls

    def _setup_layers(self):
        """Setup required napari layers."""
        # Create hand tracking layers
        if "points_left" not in self.viewer.layers:
            self.tracking_layers["left_hand"] = self.viewer.add_points(
                ndim=3, name="points_left", size=9, face_color="blue"
            )

        if "points_right" not in self.viewer.layers:
            self.tracking_layers["right_hand"] = self.viewer.add_points(
                ndim=3, name="points_right", size=9, face_color="red"
            )

        if "points_marker" not in self.viewer.layers:
            self.tracking_layers["marker"] = self.viewer.add_points(
                ndim=3, name="points_marker", size=12, face_color="white"
            )

        if "plane_position_point" not in self.viewer.layers:
            self.tracking_layers["plane_pos"] = self.viewer.add_points(
                ndim=3, name="plane_position_point", size=12, face_color="yellow"
            )

    def _start_tracking(self):
        """Start hand tracking."""
        if self.is_tracking:
            return

        # Initialize hand tracker
        if not self.hand_tracker.initialize():
            self.status_label.setText("Error: Failed to initialize hand tracker")
            return
        
        if len(self.viewer.layers) == 0:
            self.status_label.setText("Error: Add at least one 3D image layer to the viewer")
            return

        # Start worker thread (camera initialization happens inside the thread)
        self.worker = camera_worker(
            self.hand_tracker, self._process_hand_data, self.framerate, 0  # camera_id
        )

        # Connect plane affine events
        if "plane" in self.viewer.layers:
            plane_layer = self.viewer.layers["plane"]
            plane_layer.events.affine.connect(self._plane_affine_changed)

            self.old_affine_matrix = plane_layer.affine.affine_matrix.copy()
            self.new_affine_matrix = plane_layer.affine.affine_matrix.copy()

        self.worker.start()

        # Update UI
        self.is_tracking = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Hand tracking: Running")
        
        # Enable calibration buttons
        self.capture_near_btn.setEnabled(True)
        self.capture_far_btn.setEnabled(True)
        self._update_calibration_status()
        
        # Start camera preview only if calibration tab is active
        self._update_camera_preview_state()

    def _stop_tracking(self):
        """Stop hand tracking."""
        if not self.is_tracking:
            return

        # Stop worker
        if self.worker is not None:
            self.worker.send("stop")
            self.worker.quit()
            self.worker = None

        # Cleanup
        self.hand_tracker.cleanup()

        # Stop camera preview
        self.camera_preview.stop_preview()

        # Update UI
        self.is_tracking = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Hand tracking: Stopped")
        
        # Disable calibration buttons
        self.capture_near_btn.setEnabled(False)
        self.capture_far_btn.setEnabled(False)
        self._update_calibration_status()

        # Clear tracking data
        for layer in self.tracking_layers.values():
            layer.data = np.array([]).reshape(0, 3)

    def _on_tab_changed(self, index):
        """Handle tab change to optimize resource usage."""
        self._update_camera_preview_state()

    def _update_camera_preview_state(self):
        """Update camera preview state based on current tab and tracking status."""
        calibration_tab_active = self.tab_widget.currentIndex() == 1
        should_preview = self.is_tracking and calibration_tab_active
        
        if should_preview and not self.camera_preview.preview_enabled:
            self.camera_preview.start_preview()
        elif not should_preview and self.camera_preview.preview_enabled:
            self.camera_preview.stop_preview()

    def _process_hand_data(self, results):
        """Process hand detection results and update napari layers.

        Parameters
        ----------
        results : dict
            Hand detection results
        """
        if not results:
            return

        # Update camera preview
        frame = results.get('frame')
        if frame is not None and self.tab_widget.currentIndex() == 1:  # Only update if calibration tab is active
            # Select the best hand landmarks for overlay (prefer left hand, fallback to right)
            best_hand_landmarks = None
            if results.get('left_hand_raw') is not None:
                best_hand_landmarks = results.get('left_hand_raw')
            elif results.get('right_hand_raw') is not None:
                best_hand_landmarks = results.get('right_hand_raw')
            
            self.camera_preview.update_frame(frame, best_hand_landmarks)

        # Get the actual volume shape from the first image layer
        img_shape = None
        for layer in self.viewer.layers:
            if layer.data.ndim == 3:
                img_shape = layer.data.shape
                break
        
        if img_shape is None:
            # Fallback to default shape
            img_shape = (128, 128, 128)

        left_hand = results.get("left_hand")
        right_hand = results.get("right_hand")
        
        # Store latest hand data for calibration (prefer left hand, fallback to right)
        if left_hand:
            self._latest_hand_data = {
                'landmarks_raw': results.get('left_hand_raw'),  # Raw MediaPipe landmarks
                'handedness': 'Left'
            }
        elif right_hand:
            self._latest_hand_data = {
                'landmarks_raw': results.get('right_hand_raw'),  # Raw MediaPipe landmarks  
                'handedness': 'Right'
            }
        else:
            self._latest_hand_data = None

        # Get plane layer
        plane_layer = None
        try:
            plane_layer = self.viewer.layers["plane"]
        except KeyError:
            return

        # Process left hand (plane control)
        if left_hand:
            self._process_left_hand(left_hand, img_shape, plane_layer)
        else:
            # Clear left hand points
            self.tracking_layers["left_hand"].data = np.array([]).reshape(0, 3)

        # Process right hand (marker/manipulation)
        if right_hand:
            self._process_right_hand(right_hand, img_shape, plane_layer)
        else:
            # Clear right hand points
            self.tracking_layers["right_hand"].data = np.array([]).reshape(0, 3)
            self.marking = False

        # Update plane position indicator
        if plane_layer:
            plane_pos = multiply_affine(
                plane_layer.affine.affine_matrix, plane_layer.plane.position
            )
            self.tracking_layers["plane_pos"].data = np.array([plane_pos])

    def _process_left_hand(self, left_hand, img_shape, plane_layer):
        """Process left hand for plane control."""
        landmarks = left_hand["landmarks"]
        center = left_hand["center"]
        orientation = left_hand["orientation"]

        # Apply depth calibration if available
        if self.depth_calibrator.is_calibrated() and self._latest_hand_data:
            try:
                hand_depth, landmark_depths = self.depth_calibrator.estimate_depths(
                    self._latest_hand_data['landmarks_raw'], 640, 480
                )
                # Apply depth scaling to landmarks (convert from [0,1] to actual depth)
                # Here we map depth to Z coordinate in volume space
                depth_scale = img_shape[0] * 0.5  # Half the volume depth
                landmarks = landmarks.copy()
                landmarks[:, 0] = (1.0 - landmark_depths) * depth_scale  # Invert depth (0=far, 1=near)
                
                center = center.copy()
                center[0] = (1.0 - hand_depth) * depth_scale
            except Exception as e:
                print(f"Depth calibration error: {e}")
                # Fall back to original coordinates

        # Convert to napari coordinates
        pos_center = points_coords_to_napari_coords(center, img_shape)

        # Update hand points visualization
        hand_points = (landmarks - center) * 200 + pos_center
        self.tracking_layers["left_hand"].data = hand_points

        # Update plane position and orientation
        normal = orientation[0]  # z-axis of hand reference frame

        # Transform to plane's coordinate system
        M_inv = invert_affine(plane_layer.affine.affine_matrix)
        pos_center_affine = multiply_affine(M_inv, pos_center)
        normal_affine = M_inv[:3, :3].T @ normal

        # Smooth interpolation
        old_pos = np.array(plane_layer.plane.position)
        old_normal = np.array(plane_layer.plane.normal)

        new_pos = (
            self.interp_coeff * old_pos + (1 - self.interp_coeff) * pos_center_affine
        )
        new_normal = (
            self.interp_coeff * old_normal + (1 - self.interp_coeff) * normal_affine
        )

        # Update plane
        plane_layer.plane.position = new_pos
        plane_layer.plane.normal = new_normal

    def _process_right_hand(self, right_hand, img_shape, plane_layer):
        """Process right hand for marking/manipulation."""
        landmarks = right_hand["landmarks"]
        center = right_hand["center"]
        is_holding = right_hand["is_holding"]

        # Apply depth calibration if available (same as left hand)
        if self.depth_calibrator.is_calibrated() and self._latest_hand_data:
            try:
                hand_depth, landmark_depths = self.depth_calibrator.estimate_depths(
                    self._latest_hand_data['landmarks_raw'], 640, 480
                )
                # Apply depth scaling to landmarks (convert from [0,1] to actual depth)
                # Here we map depth to Z coordinate in volume space
                depth_scale = img_shape[0] * 0.5  # Half the volume depth
                landmarks = landmarks.copy()
                landmarks[:, 0] = (1.0 - landmark_depths) * depth_scale  # Invert depth (0=far, 1=near)
                
                center = center.copy()
                center[0] = (1.0 - hand_depth) * depth_scale
            except Exception as e:
                print(f"Depth calibration error: {e}")
                # Fall back to original coordinates

        # Convert to napari coordinates
        pos_center = points_coords_to_napari_coords(center, img_shape)

        # Update hand points visualization
        hand_points = (landmarks - center) * 200 + pos_center
        self.tracking_layers["right_hand"].data = hand_points

        # Handle marking with holding gesture
        if is_holding:
            # Use thumb tip (landmark 4) for marking position
            marker_pos = points_coords_to_napari_coords(landmarks[4], img_shape)

            if self.marking:
                # Update last marker position
                if len(self.tracking_layers["marker"].data) > 0:
                    current_data = self.tracking_layers["marker"].data.copy()
                    current_data[-1] = marker_pos
                    self.tracking_layers["marker"].data = current_data
            else:
                # Add new marker
                current_data = self.tracking_layers["marker"].data
                if len(current_data) == 0:
                    new_data = marker_pos.reshape(1, -1)
                else:
                    new_data = np.vstack([current_data, marker_pos])
                self.tracking_layers["marker"].data = new_data
                self.marking = True
        else:
            self.marking = False

    def _plane_affine_changed(self, event):
        """Handle plane affine transformation changes."""
        plane_layer = event.source

        # Update affine matrices
        self.old_affine_matrix = self.new_affine_matrix.copy()
        self.new_affine_matrix = plane_layer.affine.affine_matrix.copy()

        # Transform plane position to maintain consistency
        plane_position = np.array(plane_layer.plane.position)

        plane_position = multiply_affine(
            invert_affine(self.new_affine_matrix),
            multiply_affine(self.old_affine_matrix, plane_position),
        )
        plane_layer.plane.position = plane_position

        # Transform plane normal
        plane_normal = np.array(plane_layer.plane.normal)
        plane_normal = (
            self.new_affine_matrix[:3, :3].T
            @ self.old_affine_matrix[:3, :3]
            @ plane_normal
        )
        plane_layer.plane.normal = plane_normal

    def _capture_hand_position(self, distance_type):
        """Capture hand position at specified distance.
        
        Parameters
        ----------
        distance_type : str
            Either 'Near' or 'Far' to specify the distance type
        """
        if not self.is_tracking:
            self.calib_info.setText("Error: Start tracking first")
            return
        
        # Get current hand data
        current_hand = self._get_current_hand_for_calibration()
        if current_hand is None:
            self.calib_info.setText("Error: No hand detected. Show your palm to the camera.")
            return
    
        # Capture position based on distance type
        capture_func = self.depth_calibrator.capture_near if distance_type=="Near" else self.depth_calibrator.capture_far
        success = capture_func(
            current_hand['landmarks_raw'], 640, 480  # Standard camera resolution
        )
        
        if success:
            self.calib_info.setText(f"✓ {distance_type} position captured successfully!")
        else:
            self.calib_info.setText("Error: Failed to capture near position")
        
        self._update_calibration_status()
    
    def _capture_near(self):
        """Capture hand position at near distance."""
        self._capture_hand_position('Near')
    
    def _capture_far(self):
        """Capture hand position at far distance."""
        self._capture_hand_position('Far')
    
    def _finalize_calibration(self):
        """Finalize the depth calibration."""
        success = self.depth_calibrator.finalize()
        
        if success:
            self.calib_info.setText("✓ Calibration completed successfully!\nDepth estimation is now active.")
        else:
            self.calib_info.setText("Error: Failed to finalize calibration. Check that you captured both near and far positions.")
        
        self._update_calibration_status()
    
    def _reset_calibration(self):
        """Reset the depth calibration."""
        self.depth_calibrator.reset()
        self.calib_info.setText("Calibration reset. Start with capturing near and far positions.")
        self._update_calibration_status()
    
    def _update_calibration_status(self):
        """Update calibration status display."""
        status = self.depth_calibrator.get_status()
        self.calib_status_label.setText(f"Status: {status}")
        
        # Update button states
        has_near = self.depth_calibrator.near is not None
        has_far = self.depth_calibrator.far is not None
        is_calibrated = self.depth_calibrator.is_calibrated()
        
        self.finalize_btn.setEnabled(has_near and has_far and not is_calibrated)
    
    def _get_current_hand_for_calibration(self):
        """Get current hand data for calibration (prefers left hand)."""
        # This would be populated by the tracking system
        # For now, return None - this needs to be connected to the hand tracker
        # In a full implementation, you'd store the latest hand detection results
        return getattr(self, '_latest_hand_data', None)
