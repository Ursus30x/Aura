# Aura Project

Aura is a real-time facial animation system that bridges Computer Vision (Python) with high-performance Graphics (C++). It uses MediaPipe for face tracking and ZeroMQ to stream blendshape data to an OpenGL renderer.

## Requirements

### System Dependencies (Linux/Ubuntu)

Install the necessary C++ build tools and libraries (GLFW, ZeroMQ, nlohmann-json):

```bash
sudo apt update
sudo apt install build-essential cmake libglfw3-dev libzmq3-dev nlohmann-json3-dev python3-venv python3-pip
```

### System Dependencies (Arch Linux)

Install the necessary C++ build tools and libraries:

```bash
sudo pacman -Syu
sudo pacman -S cmake glfw-x11 glm zeromq python-pip python-venv
```

### Python Dependencies

The Computer Vision module requires Python 3.8+ and the following packages:

*   `opencv-python`
*   `mediapipe`
*   `pyzmq`

You can install them via pip:

```bash
pip install opencv-python mediapipe pyzmq
```

## Build Instructions

### C++ Graphics Module

1.  Create a build directory in the project root:
    ```bash
    mkdir buildff
	
    cd build
    ```

2.  Generate build files and compile:
    ```bash
    cmake ..
    make
    ```

## How to Run

1.  **Start the CV Module (Publisher):**
    Navigate to the `cv` directory and run the face mesh application with a video file.
    ```bash
    python cv/src/face_mesh_app.py path/to/video.mp4
    ```

2.  **Start the Graphics Module (Subscriber):**
    Run the compiled C++ executable from the `build` directory, providing the paths to your OBJ mesh files (Base mesh + 5 morph targets).
    ```bash
    ./build/App base.obj mouth.obj l_full.obj l_half.obj r_full.obj r_half.obj
    ```