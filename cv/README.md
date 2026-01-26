# Face Mesh & Blendshapes Module

This module performs real-time face landmark detection and blendshape analysis using MediaPipe. It processes a video input, visualizes the results (including eye openness and mouth openness), and publishes the analysis data via ZeroMQ for consumption by other modules (e.g., animation engines).

## Prerequisites

*   Python 3.8+
*   A video file (mp4) for testing.

## Installation

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Start the Face Mesh Application

Run the main script providing the path to your video file:

```bash
python face_mesh_app.py path/to/your/video.mp4
```

This will:
*   Open a window showing the video with landmarks and metrics overlay.
*   Start a ZeroMQ publisher on `tcp://*:5555`.

**Controls:**
*   `q`: Quit the application.
*   `SPACE`: Pause/Resume the video.
*   `b`: Print current blendshape values to the console.
*   `d`: Toggle debug mode (visualize all face landmarks).

### 2. Start the Data Listener (Optional)

To verify that data is being sent correctly, you can run the listener script in a separate terminal while the main app is running:

```bash
python zmq_listener.py
```

This script subscribes to the ZeroMQ publisher and prints the received JSON data to the console.