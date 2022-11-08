# Struct-MDC 


<div align="center">
  
  [![video](https://img.shields.io/badge/YouTube-B31B1B.svg)]()
  
  [![journal](https://img.shields.io/badge/RA_L-9672726-4b44ce.svg)](https://ieeexplore.ieee.org/document/9767763?source=authoralert)
  [![arxiv](https://img.shields.io/badge/arXiv-2204.13877-B31B1B.svg)](https://arxiv.org/abs/2204.13877)

</div>

*(click the above buttons for redirection!)*

***
#### Official page of "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM", which is accepted in IEEE RA-L'22 & IROS'22
- Depth completion from Visual(-inertial) SLAM using point & line features.

#### Code & Dataset verion info.  
- Code (including source code, utility code for visualization) & Dataset will be finalized & released soon! 
- version info
  - (04/20) docker image has been uploaded.
  - (04/21) Dataset has been uploaded.
  - (04/21) Visusal-SLAM module (modified [UV-SLAM](https://github.com/url-kaist/UV-SLAM)) has been uploaded.
  - (11/01) data generation tools has been uploaded.
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
**<span style="color:red"> (You can skip the Visual-SLAM module part, if you just want to use NYUv2, VOID, and PLAD datsets)</span>**
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





<br><br>
## Download (processed-) Datasets & Pre-trained weights
- **Datasets**
  - There are three datasets we used for verifying the performance of our proposed method.
  - We kindly share our modified datasets, which include also the `line feature` from [UV-SLAM](https://github.com/url-kaist/UV-SLAM) and already has been pre-processed. For testing our method, please use the `modified` one.

- **Pre-trained weights**
  - We provide our pre-trained network, which is same as the one used in the paper.
  - There are two files for each dataset: pre-trained weights for 
    - **Depth network:** used at training/evaluation time.
    - **Pose network:** used only at training time. (for supervision)

- **PLAD**
  - This is our proposed dataset, which has `point & line feature depth` from [UV-SLAM](https://github.com/url-kaist/UV-SLAM).
  - Each sequence was acquired in various indoor / outdoor man-made environment. 
  - We also distribute the improved weights with original weights used for the paper. For each evaluation result, please refer [results]().


For more details on each dataset we used, please refer our [paper](https)

<div align="center">
  
| Dataset | ref. link | train/eval data | ROS bag | DepthNet weight | PoseNet weight |
|:-------:|:-------:|:-------:|:--------------:|:--------------:|:--------------:|
|   VOID  | [original](https://github.com/alexklwong/void-dataset) |     [void-download](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/VOID/void_parsed_line.tar.xz) | [void-raw](https://github.com/alexklwong/void-dataset#downloading-void) | [void-depth](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_VOID/my/structMDC_model/depth_model-best.pth) | [void-pose](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_VOID/my/structMDC_model/depth_model-best.pth) |
|  NYUv2  | [pre-processed](https://github.com/fangchangma/sparse-to-dense) |     [nyu-download](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/NYUv2/nyu_v2_line.tar.xz) | - | [nyu-depth](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_NYUv2/my/structMDC_model/depth_model-best.pth) | [nyu-pose](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_NYUv2/my/structMDC_model/pose_model-best.pth) |
|   PLAD  | proposed! |     [plad-download](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/PLAD/PLAD_v2.tar.xz) | [plad-raw](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/PLAD/PLAD_raw.tar.xz) |  [plad-depth (paper ver.)](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_PLAD/my/0321_low_error/structMDC_model/depth_model-best.pth)<br>[plad-depth (improved ver.)](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_PLAD/my/0317_low_error_(IROS_retrained)/depth_model-best.pth) | [plad-pose (paper ver.)](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_PLAD/my/0321_low_error/structMDC_model/pose_model-best.pth)<br>[plad-pose (improved ver.)](https://urserver.kaist.ac.kr/publicdata/PLAD_Struct-MDC/jinwoo_from_MIL1/evaluation_results/main_exp_PLAD/my/0317_low_error_(IROS_retrained)/pose_model-best.pth) |

</div>








<br><br>
## Running Struct-MDC

Using our pre-trained network, you can simply run our network and verify the performance as same as our paper.
```bash
# move the pre-trained weights to the following folder:
cd Struct-MDC/Struct-MDC_src/
mv $(PATH_TO_PRETRAINED_WEIGHTS)  ./pretrained/{plad | nyu | void}/struct_MDC_model/

# link dataset
tar -xvzf {PLAD: PLAD_v2 | VOID: void_parsed_line | NYUV2: nyu_v2_line}.tar.xz
ln -s $(PATH_TO_DATASET_FOLDER) ./data/

# running
bash bash/run_structMDC_{plad | nyu | void}_pretrain.sh
```

- **Evaluation results**

<div align="center">
  
|              | **MAE**  | **RMSE** | **< 1.05** | **< 1.10** | **< 1.25^3** |
|:------------:|----------|----------|------------|:----------:|:------------:|
|   **paper**  | 1170.303 | 1481.583 | 4.567      |    8.899   |    67.071    |
| **improved** | 1142.019  | 1411.193  | 3.945  | 7.785   |    67.617    |

</div>                                                             
                                                             
  
<br><br>
## Training Struct-MDC
You can also train the network from the beginning using your own data. <br>
However, in this case, you have to prepare the data as same as ours, from following procedure: <br>

*(Since our data structure, dataloader, and pre-processing code templetes follows our baseline: [KBNet](https://github.com/alexklwong/calibrated-backprojection-network), you can also refer the author's link. We thanks the KBNet's authors.)*

```bash
# data preparation
cd Struct-MDC_src
mkdir data
ln -s $(PATH_TO_{NYU | VOID | PLAD}_DATASET_ROOT) data/
python3 setup/setup_dataset_{nyu | void | plad}.py

# train
bash bash/train_{nyu | void | plad}.sh
```

<br><br>
## Prepare Your own Dataset
- *Only if you want to make your own dataset from UV-SLAM depth estimation*, follow [Post-process your own Dataset](https://github.com/url-kaist/Struct-MDC/tree/main/data_generation#readme).






<br><br>
## Citation
If you use the algorithm in an academic context, please cite the following publication:
```
@article{jeon2022struct,
  title={Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities From Visual SLAM},
  author={Jeon, Jinwoo and Lim, Hyunjun and Seo, Dong-Uk and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={3},
  pages={6391--6398},
  year={2022},
  publisher={IEEE}
}

@article{lim2022uv,
  title={UV-SLAM: Unconstrained Line-based SLAM Using Vanishing Points for Structural Mapping},
  author={Lim, Hyunjun and Jeon, Jinwoo and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE},
  volume={7},
  number={2},
  pages={1518-1525},
  doi={10.1109/LRA.2022.3140816}
}
```




<br><br>
## Acknowledgements
- **Visual-SLAM module** <br>
We use [UV-SLAM](https://github.com/url-kaist/UV-SLAM), which is based on [VINS-MONO](https://github.com/HKUST-Aerial-Robotics/VINS-Mono), as our baseline code. Thanks for H. Lim and Dr. Qin Tong, Prof. Shen etc very much. 
<!--
This work was financially supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00230, development of real·virtual environmental analysis based adaptive interaction technology) and in part by the Defense Challengeable Future Technology Program of Agency for Defense Development, Republic of Korea. The students are supported by Korea Ministry of Land, Infrastructure and Transport (MOLIT) as "Innovative Talent Education Program for Smart City" and BK21 FOUR.
-->

- **Depth completion module** <br>
We use [KBNet](https://github.com/alexklwong/calibrated-backprojection-network) as our baseline code. Thanks for W. Alex and S. Stefano very much.





<br><br>
## Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
We are still working on improving the code reliability.
For any technical issues, please contact Jinwoo Jeon (zinuok@kaist.ac.kr).
