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
Author: Alex Wong <alexw@cs.ucla.edu> (original: train_kbnet.py)

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
import argparse
import torch
import global_constants as settings
from structMDC import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_mesh_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_validity_map_path',
    type=str, required=True, help='Path to list of training validity map paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Path to list of training camera intrinsics paths')

parser.add_argument('--val_image_path',
    type=str, default='', help='Path to list of validation image paths')
parser.add_argument('--val_mesh_depth_path',
    type=str, default='', help='Path to list of validation sparse depth paths')
parser.add_argument('--val_validity_map_path',
    type=str, required=True, help='Path to list of validation validity map paths')
parser.add_argument('--val_intrinsics_path',
    type=str, default='', help='Path to list of validation camera intrinsics paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Path to list of validation ground truth depth paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')

# Data parameters
parser.add_argument('--cropped_margin',
    type=int, default=settings.CROP_MARGIN, help='cropped margin of each side')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=settings.INPUT_CHANNELS_IMAGE, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=settings.INPUT_CHANNELS_DEPTH, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=settings.NORMALIZED_IMAGE_RANGE, help='Range of image intensities after normalization')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=settings.OUTLIER_REMOVAL_KERNEL_SIZE, help='Kernel size to filter outlier sparse depth')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=settings.OUTLIER_REMOVAL_THRESHOLD, help='Difference threshold to consider a point an outlier')
# Mesh Depth Refinement settings
parser.add_argument('--n_convolution_mesh_depth_refine',
    type=int, default=settings.N_CONVOLUTION_MESH_DEPTH_REFINE, help='Number of convolutions for mesh depth refinement')
parser.add_argument('--n_filter_mesh_depth_refine',
    type=int, default=settings.N_FILTER_MESH_DEPTH_REFINE, help='Number of filters for mesh depth refinement')
# Depth network settings
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_IMAGE, help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_DEPTH, help='Space delimited list of filters to use in each block of depth encoder')
parser.add_argument('--resolutions_backprojection',
    nargs='+', type=int, default=settings.RESOLUTIONS_BACKPROJECTION, help='Space delimited list of resolutions to use calibrated backprojection')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER, help='Space delimited list of filters to use in each block of depth decoder')
parser.add_argument('--deconv_type',
    type=str, default=settings.DECONV_TYPE, help='Deconvolution type: up, transpose')
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum value of predicted depth')
# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default=settings.WEIGHT_INITIALIZER, help='Weight initialization type: kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function after each layer: relu, leaky_relu, elu, sigmoid')
# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')
# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=settings.AUGMENTATION_PROBABILITIES, help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=settings.AUGMENTATION_SCHEDULE, help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=settings.AUGMENTATION_RANDOM_CROP_TYPE, help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=settings.AUGMENTATION_RANDOM_FLIP_TYPE, help='Random flip type for data augmentation: horizontal, vertical')
parser.add_argument('--augmentation_random_remove_points',
    nargs='+', type=float, default=settings.AUGMENTATION_RANDOM_REMOVE_POINTS, help='If set, randomly remove points from sparse depth')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default=settings.AUGMENTATION_RANDOM_NOISE_TYPE, help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')
# Loss function settings
parser.add_argument('--w_color',
    type=float, default=settings.W_COLOR, help='Weight of color consistency loss')
parser.add_argument('--w_structure',
    type=float, default=settings.W_STRUCTURE, help='Weight of structural consistency loss')
parser.add_argument('--w_sparse_depth',
    type=float, default=settings.W_SPARSE_DEPTH, help='Weight of sparse depth consistency loss')
parser.add_argument('--w_smoothness',
    type=float, default=settings.W_SMOOTHNESS, help='Weight of local smoothness loss')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=settings.W_WEIGHT_DECAY_DEPTH, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=settings.W_WEIGHT_DECAY_POSE, help='Weight of weight decay regularization for pose')
# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum value of depth to evaluate')
# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of iterations for each checkpoint')
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of iterations before logging summary')
parser.add_argument('--n_summary_display',
    type=int, default=settings.N_SUMMARY_DISPLAY, help='Number of samples to include in visual display summary')
parser.add_argument('--validation_start_step',
    type=int, default=settings.VALIDATION_START_STEP, help='Number of steps before starting validation')
parser.add_argument('--depth_model_restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore depth model from checkpoint')
parser.add_argument('--pose_model_restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore pose model from checkpoint')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    args.val_image_path = None if args.val_image_path == '' else args.val_image_path
    args.val_mesh_depth_path = None if args.val_mesh_depth_path == '' else args.val_mesh_depth_path
    args.val_validity_map_path = None if args.val_validity_map_path == '' else args.val_validity_map_path
    args.val_intrinsics_path = None if args.val_intrinsics_path == '' else args.val_intrinsics_path
    args.val_ground_truth_path = None if args.val_ground_truth_path == '' else args.val_ground_truth_path

    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    # Checkpoint settings
    args.depth_model_restore_path = None if args.depth_model_restore_path == '' else args.depth_model_restore_path

    args.pose_model_restore_path = None if args.pose_model_restore_path == '' else args.pose_model_restore_path

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in settings.DEVICE_AVAILABLE:
        args.device = settings.CUDA if torch.cuda.is_available() else settings.CPU

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    train(train_image_path=args.train_image_path,
          train_mesh_depth_path=args.train_mesh_depth_path,
          train_validity_map_path=args.train_validity_map_path,
          train_intrinsics_path=args.train_intrinsics_path,
          val_image_path=args.val_image_path,
          val_mesh_depth_path=args.val_mesh_depth_path,
          val_validity_map_path=args.val_validity_map_path,
          val_intrinsics_path=args.val_intrinsics_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          # Data settings
          cropped_margin=args.cropped_margin,
          # Input settings
          input_channels_image=args.input_channels_image,
          input_channels_depth=args.input_channels_depth,
          normalized_image_range=args.normalized_image_range,
          outlier_removal_kernel_size=args.outlier_removal_kernel_size,
          outlier_removal_threshold=args.outlier_removal_threshold,
          # Mesh Depth Refinement settings
          n_convolution_mesh_depth_refine=args.n_convolution_mesh_depth_refine,
          n_filter_mesh_depth_refine=args.n_filter_mesh_depth_refine,
          # Depth network settings
          n_filters_encoder_image=args.n_filters_encoder_image,
          n_filters_encoder_depth=args.n_filters_encoder_depth,
          resolutions_backprojection=args.resolutions_backprojection,
          n_filters_decoder=args.n_filters_decoder,
          deconv_type=args.deconv_type,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          # Augmentation settings
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          augmentation_random_remove_points=args.augmentation_random_remove_points,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          # Loss function settings
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_sparse_depth=args.w_sparse_depth,
          w_smoothness=args.w_smoothness,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          validation_start_step=args.validation_start_step,
          depth_model_restore_path=args.depth_model_restore_path,
          pose_model_restore_path=args.pose_model_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
