"""
The solution I would like to implement is the following: prompt the user to put its hand close to the camera (the hand is fully opened, the palm is shown), press a button, then very far from the camera, press a button. Each time, the positon of the landmarks is stored. I need to define a metric based on the distance between the landmarks to get e.g a scalar (if possible make it robust to the fact that the user might not show its palm perfectly parallel to the plane of the screen on which the camera lies, perhaps by assuming a standard hand geometry). These hand visible sizes will then be used to infer an average Z for the whole hand by mapping 0 to the size when the hand was farthest, and 1 when it is closest. Then, use the z attribute of mediapipe landmarks to refine the absolute z estimation for each landmark. Could you provide a very concise example of that ?

Got it. Below is a compact, drop-in pattern that does what you described:
>one button for NEAR capture, one for FAR capture (open palm),
>a robust hand size scalar using median of 3D pairwise distances among stable palm landmarks (rotation-invariant; uses MediaPipe’s relative z to reduce foreshortening),
>map current size to [0,1] depth scalar,
>then refine per-landmark depth using MediaPipe’s relative z with a scale learned from the two calibration snapshots.
"""

import numpy as np
from itertools import combinations

# Stable palm landmarks: wrist + MCPs (less pose variability than fingertips)
PALM_IDS = [0, 5, 9, 13, 17]  # wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp

def _lm_zyx_px(hand_landmarks, img_w, img_h):
    """Return Nx3 array of landmarks in pixel units. z is scaled like x (by img_w)."""
    lms = hand_landmarks.landmark
    zyx = np.array([[lm.x*img_w, lm.y*img_h, lm.z*img_w] for lm in lms], dtype=np.float32)
    return zyx

def hand_size_metric_zyx(zyx, ids=PALM_IDS):
    """Robust size metric: median of all 3D pairwise distances over palm landmarks."""
    pts = zyx[ids]
    dists = [np.linalg.norm(pts[i]-pts[j]) for i,j in combinations(range(len(ids)), 2)]
    return float(np.median(dists))

def relative_z_offsets(hand_landmarks):
    """Return per-landmark z offsets relative to the hand mean (unitless, MediaPipe scale)."""
    zs = np.array([lm.z for lm in hand_landmarks.landmark], dtype=np.float32)
    return zs - zs.mean()

class DepthCalibrator:
    def __init__(self):
        self.near = None
        self.far = None
        self.calib = None

    def capture_near(self, hand_landmarks, img_w, img_h):
        zyx = _lm_zyx_px(hand_landmarks, img_w, img_h)
        self.near = {
            "size": hand_size_metric_zyx(zyx),
            "zoff": relative_z_offsets(hand_landmarks)
        }

    def capture_far(self, hand_landmarks, img_w, img_h):
        zyx = _lm_zyx_px(hand_landmarks, img_w, img_h)
        self.far = {
            "size": hand_size_metric_zyx(zyx),
            "zoff": relative_z_offsets(hand_landmarks)
        }

    def finalize(self):
        """Prepare calibration constants."""
        assert self.near and self.far, "Capture near and far first."
        S_near, S_far = self.near["size"], self.far["size"]

        # Scale for mapping current size -> scalar depth in [0,1]
        self.calib = {
            "S_near": S_near,
            "S_far": S_far,
        }

        # Learn a conversion for MediaPipe z offsets -> depth refinement.
        # We map the observed change of z-offset spread between far and near
        # onto the unit depth range [0,1].
        # Use a robust spread (median absolute deviation).
        def spread(zoff): return float(np.median(np.abs(zoff - np.median(zoff))))
        spread_near = spread(self.near["zoff"])
        spread_far  = spread(self.far["zoff"])
        dz_spread = max(abs(spread_near - spread_far), 1e-6)
        # One unit of z-offset spread corresponds to the whole [0,1] depth span:
        self.calib["z_to_depth_scale"] = 1.0 / dz_spread
        # Centering is relative to per-frame mean, so no intercept needed.

    def estimate_depths(self, hand_landmarks, img_w, img_h):
        """Return:
           - hand_depth_scalar in [0,1]
           - per-landmark absolute-ish depth scalars in [0,1] refined by z
        """
        assert self.calib is not None, "Run finalize() after both captures."

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


