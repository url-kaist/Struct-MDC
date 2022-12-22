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
Author: Alex Wong <alexw@cs.ucla.edu> (original: run_kbnet.py)

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
import global_constants as settings
from structMDC import run


parser = argparse.ArgumentParser()

parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--mesh_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--validity_map_path',
    type=str, required=True, help='Path to list of validity map paths')
parser.add_argument('--intrinsics_path',
    type=str, required=True, help='Path to list of camera intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Path to list of ground truth depth paths')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=7, help='Kernel size to filter outlier sparse depth')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=1.5, help='Difference threshold to consider a point an outlier')
parser.add_argument('--cropped_margin',
    type=int, default=settings.CROP_MARGIN, help='cropped margin of each side')

# Mesh Depth Refinement settings
parser.add_argument('--n_convolution_mesh_depth_refine',
    type=int, default=3, help='Number of convolutions for sparse to dense pooling')
parser.add_argument('--n_filter_mesh_depth_refine',
    type=int, default=8, help='Number of filters for sparse to dense pooling')

# Depth network settings
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=[48, 96, 192, 384, 384], help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=[16, 32, 64, 128, 128], help='Space delimited list of filters to use in each block of depth encoder')
parser.add_argument('--resolutions_backprojection',
    nargs='+', type=int, default=[0, 1, 2, 3], help='Space delimited list of resolutions to use calibrated backprojection')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 128, 64, 12], help='Space delimited list of filters to use in each block of depth decoder')
parser.add_argument('--deconv_type',
    type=str, default='up', help='Deconvolution type: up, transpose')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum value of predicted depth')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Initialization for weights')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function after each layer')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.0, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--output_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then store inputs and outputs into output path')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')
parser.add_argument('--depth_model_restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore depth model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    '''
    Assert inputs
    '''
    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    # Checkpoint settings
    args.depth_model_restore_path = None if args.depth_model_restore_path == '' else args.depth_model_restore_path

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(args.image_path,
        args.mesh_depth_path,
        args.validity_map_path,
        args.intrinsics_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        cropped_margin=args.cropped_margin,
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
        normalized_image_range=args.normalized_image_range,
        outlier_removal_kernel_size=args.outlier_removal_kernel_size,
        outlier_removal_threshold=args.outlier_removal_threshold,
        # Mesh Depth Refinement settings
        n_convolution_mesh_depth_refine_pool=args.n_convolution_mesh_depth_refine,
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
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path=args.output_path,
        depth_model_restore_path=args.depth_model_restore_path,
        # Output settings
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
