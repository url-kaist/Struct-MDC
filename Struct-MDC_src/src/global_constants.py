######################################
#     modified by Jinwoo Jeon        #
######################################
'''
Author: Jinwoo Jeon <zinuok@kaist.ac.kr>

If you use this code, please cite the following paper:
J. Jeon, H. Lim, D. Seo, H. Myung. "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM"

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
'''

######################################
#     original License info.         #
######################################
'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
# Batch settings
N_BATCH                                     = 8
N_HEIGHT                                    = 320
N_WIDTH                                     = 768

# Data settings
CROP_MARGIN                                 = 0


# Input settings
INPUT_CHANNELS_IMAGE                        = 3
INPUT_CHANNELS_DEPTH                        = 2
NORMALIZED_IMAGE_RANGE                      = [0, 1]
OUTLIER_REMOVAL_KERNEL_SIZE                 = 7
OUTLIER_REMOVAL_THRESHOLD                   = 1.5

# Sparse to dense pool settings
N_CONVOLUTION_MESH_DEPTH_REFINE             = 3
N_FILTER_MESH_DEPTH_REFINE                  = 8

# Depth network settings
N_FILTERS_ENCODER_IMAGE                     = [48, 96, 192, 384, 384]
N_FILTERS_ENCODER_DEPTH                     = [16, 32, 64, 128, 128]
RESOLUTIONS_BACKPROJECTION                  = [0, 1, 2, 3]
N_FILTERS_DECODER                           = [256, 128, 128, 64, 12]
DECONV_TYPE                                 = 'up'
MIN_PREDICT_DEPTH                           = 1.5
MAX_PREDICT_DEPTH                           = 100.0

# Weight settings
WEIGHT_INITIALIZER                          = 'xavier_normal'
ACTIVATION_FUNC                             = 'leaky_relu'

# Training settings
LEARNING_RATES                              = [5e-5, 1e-4, 15e-5, 1e-4, 5e-5, 2e-5, 1e-5]
LEARNING_SCHEDULE                           = [2, 8, 20, 30, 45, 60, 80]
AUGMENTATION_PROBABILITIES                  = [1.00, 0.50, 0.25]
AUGMENTATION_SCHEDULE                       = [50, 55, 60]
AUGMENTATION_RANDOM_CROP_TYPE               = ['horizontal', 'vertical', 'anchored', 'bottom']
AUGMENTATION_RANDOM_FLIP_TYPE               = ['none']
AUGMENTATION_RANDOM_REMOVE_POINTS           = [0.60, 0.70]
AUGMENTATION_RANDOM_NOISE_TYPE              = 'none'
AUGMENTATION_RANDOM_NOISE_SPREAD            = -1

# Loss function settings
W_COLOR                                     = 0.15
W_STRUCTURE                                 = 0.95
W_SPARSE_DEPTH                              = 0.60
W_SMOOTHNESS                                = 0.04
W_WEIGHT_DECAY_DEPTH                        = 0.00
W_WEIGHT_DECAY_POSE                         = 0.00

# Evaluation settings
MIN_EVALUATE_DEPTH                          = 0.00
MAX_EVALUATE_DEPTH                          = 100.0

# Checkpoint settings
CHECKPOINT_PATH                             = 'trained_kbnet'
N_CHECKPOINT                                = 5000
N_SUMMARY                                   = 5000
N_SUMMARY_DISPLAY                           = 4
VALIDATION_START_STEP                       = 200000
RESTORE_PATH                                = None

# Hardware settings
CUDA                                        = 'cuda'
CPU                                         = 'cpu'
GPU                                         = 'gpu'
DEVICE                                      = 'cuda'
DEVICE_AVAILABLE                            = [CPU, CUDA, GPU]
N_THREAD                                    = 8
