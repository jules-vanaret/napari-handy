"""Depth calibration for improved Z-position estimation.

This module provides depth calibration functionality that works with napari's ZYX 
coordinate convention. The calibration process:

1. Captures hand at near and far distances
2. Computes a robust hand size metric using palm landmarks
3. Maps current hand size to depth range [0,1] where 0=far, 1=near
4. Refines per-landmark depth using MediaPipe's relative Z coordinates

All coordinates follow napari's ZYX convention:
- Z: depth (0th index) - the calibrated dimension
- Y: vertical (1st index) 
- X: horizontal (2nd index)
"""

import numpy as np
from itertools import combinations
from typing import Optional, Dict, Tuple, Any


# Stable palm landmarks: wrist + MCPs (less pose variability than fingertips)
PALM_IDS = [0, 5, 9, 13, 17]  # wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp


def _lm_zyx_px(hand_landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Return Nx3 array of landmarks in pixel units in ZYX napari convention.
    
    Note: Following napari's ZYX convention to match the rest of the plugin.
    Z is scaled like X (by img_w) to maintain aspect ratio consistency.
    """
    lms = hand_landmarks.landmark
    zyx = np.array([
        [lm.z * img_w, lm.y * img_h, lm.x * img_w] 
        for lm in lms
    ], dtype=np.float32)
    return zyx


def hand_size_metric_zyx(zyx: np.ndarray, ids=PALM_IDS) -> float:
    """Robust size metric: median of all 3D pairwise distances over palm landmarks.
    
    Parameters
    ----------
    zyx : np.ndarray
        Landmark coordinates in ZYX napari convention
    ids : list
        Indices of palm landmarks to use
        
    Returns
    -------
    float
        Median pairwise distance (robust size metric)
    """
    pts = zyx[ids]
    dists = [
        np.linalg.norm(pts[i] - pts[j]) 
        for i, j in combinations(range(len(ids)), 2)
    ]
    return float(np.median(dists))


def relative_z_offsets(hand_landmarks) -> np.ndarray:
    """Return per-landmark z offsets relative to the hand mean (unitless, MediaPipe scale)."""
    zs = np.array([lm.z for lm in hand_landmarks.landmark], dtype=np.float32)
    return zs - zs.mean()


class DepthCalibrator:
    """Calibrates depth estimation by capturing hand at near and far distances."""
    
    def __init__(self):
        self.near: Optional[Dict[str, Any]] = None
        self.far: Optional[Dict[str, Any]] = None
        self.calib: Optional[Dict[str, float]] = None
    
    def capture_near(self, hand_landmarks, img_w: int, img_h: int) -> bool:
        """Capture hand at near distance.
        
        Parameters
        ----------
        hand_landmarks : mediapipe hand landmarks
            Hand landmarks from MediaPipe
        img_w : int
            Image width
        img_h : int
            Image height
            
        Returns
        -------
        bool
            True if capture successful
        """
        try:
            zyx = _lm_zyx_px(hand_landmarks, img_w, img_h)
            self.near = {
                "size": hand_size_metric_zyx(zyx),
                "zoff": relative_z_offsets(hand_landmarks)
            }
            return True
        except Exception as e:
            print(f"Failed to capture near: {e}")
            return False
    
    def capture_far(self, hand_landmarks, img_w: int, img_h: int) -> bool:
        """Capture hand at far distance.
        
        Parameters
        ----------
        hand_landmarks : mediapipe hand landmarks
            Hand landmarks from MediaPipe
        img_w : int
            Image width
        img_h : int
            Image height
            
        Returns
        -------
        bool
            True if capture successful
        """
        try:
            zyx = _lm_zyx_px(hand_landmarks, img_w, img_h)
            self.far = {
                "size": hand_size_metric_zyx(zyx),
                "zoff": relative_z_offsets(hand_landmarks)
            }
            return True
        except Exception as e:
            print(f"Failed to capture far: {e}")
            return False
    
    def finalize(self) -> bool:
        """Prepare calibration constants.
        
        Returns
        -------
        bool
            True if calibration successful
        """
        if not (self.near and self.far):
            print("Error: Need both near and far captures before finalizing")
            return False
        
        try:
            S_near, S_far = self.near["size"], self.far["size"]
            
            # Ensure near hand is actually larger than far hand
            if S_near <= S_far:
                print("Warning: Near hand should be larger than far hand")
                return False
            
            # Scale for mapping current size -> scalar depth in [0,1]
            self.calib = {
                "S_near": S_near,
                "S_far": S_far,
            }
            
            # Learn a conversion for MediaPipe z offsets -> depth refinement.
            # We map the observed change of z-offset spread between far and near
            # onto the unit depth range [0,1].
            # Use a robust spread (median absolute deviation).
            def spread(zoff): 
                return float(np.median(np.abs(zoff - np.median(zoff))))
            
            spread_near = spread(self.near["zoff"])
            spread_far = spread(self.far["zoff"])
            dz_spread = max(abs(spread_near - spread_far), 1e-6)
            
            # One unit of z-offset spread corresponds to the whole [0,1] depth span:
            self.calib["z_to_depth_scale"] = 1.0 / dz_spread
            
            return True
        except Exception as e:
            print(f"Failed to finalize calibration: {e}")
            return False
    
    def estimate_depths(self, hand_landmarks, img_w: int, img_h: int) -> Tuple[float, np.ndarray]:
        """Estimate depth values for hand.
        
        Parameters
        ----------
        hand_landmarks : mediapipe hand landmarks
            Hand landmarks from MediaPipe
        img_w : int
            Image width
        img_h : int
            Image height
            
        Returns
        -------
        Tuple[float, np.ndarray]
            - hand_depth_scalar in [0,1] (0=far, 1=near)
            - per-landmark absolute depth scalars in [0,1] refined by z
        """
        if self.calib is None:
            raise ValueError("Run finalize() after both captures.")
        
        zyx = _lm_zyx_px(hand_landmarks, img_w, img_h)
        S = hand_size_metric_zyx(zyx)
        
        # Map size -> [0,1] (clip for safety)
        Sf, Sn = self.calib["S_far"], self.calib["S_near"]
        hand_depth = np.clip((S - Sf) / max(Sn - Sf, 1e-6), 0.0, 1.0)
        
        # Refine with MediaPipe z offsets (relative within the hand)
        zoff = relative_z_offsets(hand_landmarks)
        k = self.calib["z_to_depth_scale"]
        
        # Normalize offsets to a bounded contribution (avoid overshoot on extreme poses)
        # Here we softly cap: contribute up to +/- 0.2 around the hand depth.
        refine = np.clip(k * zoff, -0.2, 0.2)
        
        per_landmark_depth = np.clip(hand_depth + refine, 0.0, 1.0)
        return hand_depth, per_landmark_depth
    
    def apply_calibrated_depth(self, landmarks_zyx: np.ndarray, img_shape: tuple) -> np.ndarray:
        """Apply calibrated depth to hand landmarks in ZYX napari coordinates.
        
        Parameters
        ----------
        landmarks_zyx : np.ndarray
            Hand landmarks in ZYX napari coordinate system
        img_shape : tuple
            Shape of the volume (Z, Y, X)
            
        Returns
        -------
        np.ndarray
            Landmarks with calibrated depth applied
        """
        if not self.is_calibrated():
            return landmarks_zyx  # Return unchanged if not calibrated
        
        # Estimate depths using the raw MediaPipe data
        # This requires the raw landmarks - we'd need to store them
        # For now, return unchanged as we need the raw MediaPipe landmarks
        # This method would be called from the hand processing pipeline
        return landmarks_zyx
    
    def is_calibrated(self) -> bool:
        """Check if calibrator is ready to use.
        
        Returns
        -------
        bool
            True if calibrated and ready
        """
        return self.calib is not None
    
    def reset(self):
        """Reset calibration data."""
        self.near = None
        self.far = None
        self.calib = None
    
    def get_status(self) -> str:
        """Get current calibration status.
        
        Returns
        -------
        str
            Status message
        """
        if self.calib is not None:
            return "Calibrated ✓"
        elif self.near and self.far:
            return "Ready to finalize"
        elif self.near:
            return "Near captured, need far"
        elif self.far:
            return "Far captured, need near"
        else:
            return "Not calibrated"