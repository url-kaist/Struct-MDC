#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_structMDC.py \
--cropped_margin 0 \
--image_path testing/plad/plad_test_image_150.txt \
--mesh_depth_path testing/plad/plad_test_mesh_depth_150.txt \
--validity_map_path testing/plad/plad_test_validity_map_150.txt \
--intrinsics_path testing/plad/plad_test_intrinsics_150.txt \
--ground_truth_path testing/plad/plad_test_ground_truth_150.txt \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--n_convolution_mesh_depth_refine 3 \
--n_filter_mesh_depth_refine 8 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--resolutions_backprojection 0 1 2 3 \
--n_filters_decoder 256 128 128 64 12 \
--deconv_type up \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--save_outputs \
--depth_model_restore_path trained_structMDC/plad/structMDC_model/depth_model-15000.pth \
--output_path evaluation_results/plad \
--device gpu
