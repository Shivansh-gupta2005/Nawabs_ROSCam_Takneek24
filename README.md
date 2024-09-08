

# TurtleBot Edge Detection and Depth Calculation

## Overview

This repository contains code and resources for tasks involving edge detection, center detection, depth calculation, and a TurtleBot that follows the edges of a platform. These tasks are implemented using a ZED stereo camera and ROS (Robot Operating System). The main objective is to detect the edges of a white platform, calculate the depth information, and command a TurtleBot to follow the detected edges.

## Features

- **Edge Detection**: Detect the edges of a white platform using the RGB data from the ZED camera.
- **Center Detection**: Mark the center of the detected platform and extract relevant data from the center point.
- **Depth Calculation**: Calculate the depth at specific points (such as the center of a table) and across the image using the ZED camera’s depth data.
- **TurtleBot Following Edges**: Implement a behavior where the TurtleBot follows the edges detected by the camera in a given environment.

## Tasks

### 1. Edge Detection

The edge detection algorithm uses RGB data from the ZED camera to find the edges of a white platform. The detected edges are highlighted and used as a reference for further tasks.

- **Input**: RGB images from ZED camera.
- **Output**: Image with detected edges and marked boundaries.

### 2. Center Detection

Once the edges are detected, the center of the platform is computed. This can be useful for orienting the robot or calculating depth at the center point.

- **Input**: Edge-detected image.
- **Output**: Image with marked center of the platform.

### 3. Depth Calculation

Depth data is provided by the ZED camera's depth sensor. The depth at the center of the platform or other relevant points is calculated to help with navigation and environment understanding.

- **Input**: Depth data from the ZED camera.
- **Output**: Depth value at the center of the platform or at specific points of interest.

### 4. TurtleBot Following Edges

The TurtleBot is programmed to follow the detected edges of the platform. Using the depth and edge information, the bot navigates while maintaining a safe distance from the edges.

- **Input**: Edge and depth data.
- **Output**: TurtleBot follows the platform's edges autonomously.

## Getting Started

### Prerequisites

- **ROS Noetic** (or compatible ROS version)

- **TurtleBot3** (or a similar robot for testing the edge-following behavior)
- **Python 3** (for script execution)
- **OpenCV** (for image processing)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Shivansh_gupta2005/Takneek_Nawabs.git
    ```

    ```

## Contributors

Shivansh Gupta 230976
Rishabh Chandrakar 230856
Rattandeep SIngh PUar 240854
Pradyumn Vikram 240759



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adapt and expand this based on the specific details of your project.
