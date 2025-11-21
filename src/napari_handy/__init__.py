try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    HandyWidget
)
from ._depth_calibrator import DepthCalibrator
from ._camera_preview import CameraPreviewWidget

__all__ = (
    "HandyWidget",
    "DepthCalibrator",
    "CameraPreviewWidget",
)
