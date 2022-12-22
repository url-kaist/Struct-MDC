#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

python src/train_structMDC.py \
--train_image_path training/nyu/nyu_train_image_150.txt \
--train_mesh_depth_path training/nyu/nyu_train_mesh_depth_150.txt \
--train_validity_map_path training/nyu/nyu_train_validity_map_150.txt \
--train_intrinsics_path training/nyu/nyu_train_intrinsics_150.txt \
--val_image_path testing/nyu/nyu_test_image_150.txt \
--val_mesh_depth_path testing/nyu/nyu_test_mesh_depth_150.txt \
--val_validity_map_path testing/nyu/nyu_test_validity_map_150.txt \
--val_intrinsics_path testing/nyu/nyu_test_intrinsics_150.txt \
--val_ground_truth_path testing/nyu/nyu_test_ground_truth_150.txt \
--n_batch 4 \
--cropped_margin 16 \
--n_height 448 \
--n_width 608 \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--n_convolution_mesh_depth_refine 3  \
--n_filter_mesh_depth_refine 8 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--resolutions_backprojection 0 1 2 3 \
--n_filters_decoder 256 128 128 64 12 \
--deconv_type up \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 1e-4 5e-5  \
--learning_schedule 20 30 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical anchored \
--augmentation_random_remove_points 0.30 0.60 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--w_color 0.15 \
--w_structure 0.95 \
--w_sparse_depth 2.00 \
--w_smoothness 2.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_summary 1000 \
--n_checkpoint 1000 \
--validation_start 5000 \
--checkpoint_path trained_structMDC/nyu/structMDC_model \
--device gpu \
--n_thread 8
