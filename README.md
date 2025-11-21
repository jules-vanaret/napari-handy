# napari-handy

[![License BSD-3](https://img.shields.io/pypi/l/napari-handy.svg?color=green)](https://github.com/jules-vanaret/napari-handy/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-handy.svg?color=green)](https://pypi.org/project/napari-handy)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-handy.svg?color=green)](https://python.org)
[![tests](https://github.com/jules-vanaret/napari-handy/workflows/tests/badge.svg)](https://github.com/jules-vanaret/napari-handy/actions)
[![codecov](https://codecov.io/gh/jules-vanaret/napari-handy/branch/main/graph/badge.svg)](https://codecov.io/gh/jules-vanaret/napari-handy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-handy)](https://napari-hub.org/plugins/napari-handy)

**Capture your hands with a camera and control Napari layers with intuitive gestures!**

napari-handy enables touchless, gesture-based control of 3D data visualization in napari using computer vision and hand tracking.

## Features

🖐️ **Hand Gesture Control**
- Control 3D plane position and orientation with your left hand
- Mark points in 3D space using pinch gestures with your right hand
- Real-time hand landmark visualization

📹 **Camera Integration**
- Automatic camera detection and initialization
- Configurable frame rate and processing parameters
- Robust camera resource management

⚡ **Performance Optimized**
- Lazy loading of OpenCV and MediaPipe to avoid napari conflicts
- Multi-threaded processing for smooth real-time interaction
- Efficient coordinate transformations and interpolation

## Usage

### Basic Usage

1. **Install the plugin** (see Installation section below)

2. **Start napari and load your 3D data:**
   ```python
   import napari
   import numpy as np
   
   # Load your 3D volume
   viewer = napari.Viewer(ndisplay=3)
   volume = np.load('your_3d_data.npy')  # Your 3D data
   viewer.add_image(volume)
   ```

3. **Add the Hand Gesture Control widget:**
   - Go to `Plugins > napari-handy > Hand Gesture Control`
   - Or programmatically:
   ```python
   from napari_handy import HandyWidget
   widget = HandyWidget(viewer)
   viewer.window.add_dock_widget(widget, area='right')
   ```

4. **Start hand tracking:**
   - Click "Start Tracking" in the widget
   - Position yourself in front of your camera
   - Use hand gestures to control the visualization!

### Hand Gestures

**Left Hand - Plane Control:**
- Move your left hand to control the position of the cutting plane
- Rotate your hand to change the plane orientation
- The plane will smoothly follow your hand movements

**Right Hand - Point Marking:**
- Make a pinch gesture (bring thumb and index finger together) to mark points
- While pinching, move your hand to draw continuous marks
- Release the pinch to stop marking

### Parameters

- **Framerate**: Adjust processing speed (1-30 FPS)
- **Interpolation Coefficient**: Control smoothness of movements (0.01-0.99)
  - Higher values = more responsive but potentially jittery
  - Lower values = smoother but less responsive

## Demo

Run the included demo to try napari-handy with sample data:

```bash
python -m napari_handy.demo_handy
```

## Installation

### Requirements

- Python 3.10+
- A working camera (webcam, USB camera, etc.)
- OpenCV-compatible system

### Install via pip

```bash
pip install napari-handy
```

### Install with napari

If napari is not already installed:

```bash
pip install "napari-handy[all]"
```

### Development Installation

```bash
git clone https://github.com/jules-vanaret/napari-handy.git
cd napari-handy
pip install -e ".[dev]"
```

## Technical Details

### Architecture

The plugin is organized into modular components:

- **HandTracker**: MediaPipe-based hand detection and landmark extraction
- **CameraController**: Camera management with lazy OpenCV imports
- **GeometryUtils**: Coordinate transformations and hand pose calculations
- **AffineUtils**: 3D transformation matrix operations
- **HandyWidget**: Main napari widget integrating all components

### Key Features

- **Lazy Imports**: OpenCV and MediaPipe are imported only when needed to prevent conflicts with napari's Qt backend
- **Thread Safety**: Camera processing runs in a separate thread using napari's threading utilities
- **Robust Error Handling**: Graceful handling of camera initialization failures and detection errors
- **Memory Efficient**: Automatic cleanup of resources when tracking stops

## Troubleshooting

**Camera not detected:**
- Ensure your camera is connected and not being used by other applications
- Try different camera IDs if you have multiple cameras
- Check camera permissions on your system

**Performance issues:**
- Lower the framerate in the widget
- Ensure adequate lighting for hand detection
- Close other camera applications

**Import errors:**
- Make sure all dependencies are installed: `pip install opencv-python-headless mediapipe`
- Verify you're using opencv-python-headless (not opencv-python) to avoid Qt conflicts

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-handy` via [pip]:

```
pip install napari-handy
```

If napari is not already installed, you can install `napari-handy` with napari and Qt via:

```
pip install "napari-handy[all]"
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-handy" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
