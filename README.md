# Struct-MDC 


<div align="center">
  
  [![video](https://img.shields.io/badge/YouTube-B31B1B.svg)]()
  
  [![journal](https://img.shields.io/badge/RA_L-9672726-4b44ce.svg)](https://ieeexplore.ieee.org/document/9767763?source=authoralert)
  [![arxiv](https://img.shields.io/badge/arXiv-2204.13877-B31B1B.svg)](https://arxiv.org/abs/2204.13877)

</div>

*(click the above buttons for redirection!)*

***
#### Official page of "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM", which is accepted in IEEE RA-L'22 (IROS'22 are still being under-reviewed.)

- Depth completion from Visual(-inertial) SLAM using point & line features.

#### README & code & Dataset are still being edited.  
- Code (including source code, utility code for visualization) & Dataset will be finalized & released soon! **<span style="color:red"> (goal: I'm still organizing the code structure, until publish date)</span>**
- version info
  - (04/20) docker image has been uploaded.
  - (04/21) Dataset has been uploaded.
  - (04/21) Visusal-SLAM module (modified [UV-SLAM](https://github.com/url-kaist/UV-SLAM)) has been uploaded.
***

<br><br>





## Results
- **3D Depth estimation results**
  - [VOID](https://github.com/alexklwong/void-dataset) (left three columns) and [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) (right three columns)
  - detected features (top row), estimation from [baseline](https://github.com/alexklwong/calibrated-backprojection-network) (middle row) and **ours (bottom row)**

<p align="center" width="100%">
  <img src="https://user-images.githubusercontent.com/45934290/163770560-76ad4aca-8765-476c-9b6e-376c2dc384ba.png" width="800" height="370">
</p>



- **2D Depth estimation results**
  - [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

<div align="center">
  
| Ground truth | [Baseline](https://github.com/alexklwong/calibrated-backprojection-network) | Struct-MDC (Ours) |
|:------------:|----------|-------------------|
|       <img src="https://user-images.githubusercontent.com/45934290/156975254-94a069c4-6115-42c7-9f8a-72bf581e36f5.png" width="200">      | <img src="https://user-images.githubusercontent.com/45934290/156973904-6a9914da-bd66-4400-ac6b-67b2541f14a0.png" width="200">        | <img src="https://user-images.githubusercontent.com/45934290/156975719-b41dfdb4-7372-4d5a-ae1c-cc812b697df6.png" width="200">                 |
| <img src="https://user-images.githubusercontent.com/45934290/163781436-e50eb0fd-e238-40da-9a76-28dcbf3fb351.png" width="200">            |     <img src="https://user-images.githubusercontent.com/45934290/163781442-91cae55a-d2cb-4838-a4be-2c570a70c26b.png" width="200">    |         <img src="https://user-images.githubusercontent.com/45934290/163781444-e4863358-6825-4849-b880-6b4bcee5c6ef.png" width="200">         |
  
</div>



<br><br>







## Installation
### 1. Prerequisites (we've validated our code in the following environment!)
- **Common**
  -  Ubuntu 18.04
  - [ROS melodic](http://wiki.ros.org/ROS/Installation)
- **Visual-SLAM module**
  - OpenCV 3.2.0 (under 3.4.1)
  - [Ceres Solver-1.14.0](https://github.com/zinuok/VINS-Fusion#-ceres-solver-1)
  - [Eigen-3.3.9](https://github.com/zinuok/VINS-Fusion#-eigen-1)
  - [CDT library](https://github.com/artem-ogre/CDT)
    ```
    git clone https://github.com/artem-ogre/CDT.git
    cd CDT
    mkdir build && cd build
    cmake -DCDT_USE_AS_COMPILED_LIBRARY=ON -DCDT_USE_BOOST=ON ..
    cmake --build . && cmake --install .
    sudo make install
    ```
- **Depth completion module**
  - Python 3.7.7
  - PyTorch 1.5.0
  (you can easily reproduce equivalent environment using [our docker image](https://github.com/url-kaist/Struct-MDC/blob/main/README.md#2-build))

### 2. Build
- **Visual-SLAM module**
  - As visual-SLAM, we modified the [UV-SLAM](https://github.com/url-kaist/UV-SLAM), which is implemented in ROS environment.
  - make sure that your catkin workspace has following cmake args: `-DCMAKE_BUILD_TYPE=Release`
  ```
  cd ~/$(PATH_TO_YOUR_ROS_WORKSPACE)/src
  git clone --recursive https://github.com/url-kaist/Struct-MDC
  cd ..
  catkin build
  source ~/$(PATH_TO_YOUR_ROS_WORKSPACE)/devel/setup.bash
  ```

- **Depth completion module**
  - Our depth compeltion module is based on the popular Deep-Learning framework, PyTorch.
  - For your convenience, we share our environment as Docker image. We assume that you have already installed the Docker. For Docker installation, please refer [here](https://docs.docker.com/engine/install/ubuntu/)
  ```
  # pull our docker image into your local machine
  docker pull zinuok/nvidia-torch:latest
  
  # run the image mounting our source
  docker run -it --gpus "device=0" -v $(PATH_TO_YOUR_LOCAL_FOLER):/workspace zinuok/nvidia-torch:latest bash
  ```


### 3. Trouble shooting
- any issues found will be updated in this section.
- if you've found any other issues, please post it on `Issues tab`. We'll do our best to resolve your issues.


