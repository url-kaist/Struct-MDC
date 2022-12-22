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



import os, sys, glob, argparse
import multiprocessing as mp
import numpy as np
import cv2
sys.path.insert(0, 'src')
import data_utils


dataset_name = 'nyu_v2_line'
unused_img = 0

USE_MESH = True # turn off if only line is used


NYU_ROOT_DIRPATH       = os.path.join('data', dataset_name)
NYU_OUTPUT_DIRPATH     = os.path.join('data', 'nyu_structMDC')

NYU_DATA_150_DIRPATH   = os.path.join(NYU_ROOT_DIRPATH, '')


NYU_TRAIN_IMAGE_FILENAME         = 'train_image.txt'
NYU_TRAIN_MESH_DEPTH_FILENAME    = 'train_mesh_depth.txt'
NYU_TRAIN_VALIDITY_MAP_FILENAME  = 'train_validity_map.txt'
NYU_TRAIN_GROUND_TRUTH_FILENAME  = 'train_ground_truth.txt'
NYU_TRAIN_INTRINSICS_FILENAME    = 'train_intrinsics.txt'
NYU_TEST_IMAGE_FILENAME          = 'test_image.txt'
NYU_TEST_MESH_DEPTH_FILENAME     = 'test_mesh_depth.txt'
NYU_TEST_VALIDITY_MAP_FILENAME   = 'test_validity_map.txt'
NYU_TEST_GROUND_TRUTH_FILENAME   = 'test_ground_truth.txt'
NYU_TEST_INTRINSICS_FILENAME     = 'test_intrinsics.txt'

TRAIN_REFS_DIRPATH      = os.path.join('training', 'nyu')
TEST_REFS_DIRPATH       = os.path.join('testing', 'nyu')

# NYU training set 150 density
NYU_TRAIN_IMAGE_150_FILEPATH           = os.path.join(TRAIN_REFS_DIRPATH, 'nyu_train_image_150.txt')
NYU_TRAIN_MESH_DEPTH_150_FILEPATH      = os.path.join(TRAIN_REFS_DIRPATH, 'nyu_train_mesh_depth_150.txt')
NYU_TRAIN_VALIDITY_MAP_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'nyu_train_validity_map_150.txt')
NYU_TRAIN_GROUND_TRUTH_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'nyu_train_ground_truth_150.txt')
NYU_TRAIN_INTRINSICS_150_FILEPATH      = os.path.join(TRAIN_REFS_DIRPATH, 'nyu_train_intrinsics_150.txt')
# NYU testing set 150 density
NYU_TEST_IMAGE_150_FILEPATH            = os.path.join(TEST_REFS_DIRPATH, 'nyu_test_image_150.txt')
NYU_TEST_MESH_DEPTH_150_FILEPATH       = os.path.join(TEST_REFS_DIRPATH, 'nyu_test_mesh_depth_150.txt')
NYU_TEST_VALIDITY_MAP_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'nyu_test_validity_map_150.txt')
NYU_TEST_GROUND_TRUTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'nyu_test_ground_truth_150.txt')
NYU_TEST_INTRINSICS_150_FILEPATH       = os.path.join(TEST_REFS_DIRPATH, 'nyu_test_intrinsics_150.txt')
# NYU unused testing set 150 density
NYU_UNUSED_IMAGE_150_FILEPATH          = os.path.join(TEST_REFS_DIRPATH, 'nyu_unused_image_150.txt')
NYU_UNUSED_MESH_DEPTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'nyu_unused_mesh_depth_150.txt')
NYU_UNUSED_VALIDITY_MAP_150_FILEPATH   = os.path.join(TEST_REFS_DIRPATH, 'nyu_unused_validity_map_150.txt')
NYU_UNUSED_GROUND_TRUTH_150_FILEPATH   = os.path.join(TEST_REFS_DIRPATH, 'nyu_unused_ground_truth_150.txt')
NYU_UNUSED_INTRINSICS_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'nyu_unused_intrinsics_150.txt')


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            sparse depth path at time t=0,
            validity map path at time t=0,
            ground truth path at time t=0,
            boolean flag if set then create paths only
    Returns:
        str : image reference directory path
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output validity map path at time t=0
        str : output ground truth path at time t=0
    '''

    image_path1, \
        image_path0, \
        image_path2, \
        mesh_depth_path, \
        validity_map_path, \
        ground_truth_path, \
        paths_only = inputs

    if not paths_only:
        # Create image composite of triplets
        image1 = cv2.imread(image_path1)
        image0 = cv2.imread(image_path0)
        image2 = cv2.imread(image_path2)
        imagec = np.concatenate([image1, image0, image2], axis=1)

        # Get validity map
        mesh_depth, validity_map = data_utils.load_depth_with_validity_map(mesh_depth_path)

    image_refpath = os.path.join(*image_path0.split(os.sep)[2:])

    # Set output paths
    image_outpath = os.path.join(NYU_OUTPUT_DIRPATH, image_refpath)
    mesh_depth_outpath = mesh_depth_path
    validity_map_outpath = validity_map_path
    ground_truth_outpath = ground_truth_path

    # Verify that all filenames match
    image_out_dirpath, image_filename = os.path.split(image_outpath)
    mesh_depth_filename = os.path.basename(mesh_depth_outpath)
    validity_map_filename = os.path.basename(validity_map_outpath)
    ground_truth_filename = os.path.basename(ground_truth_outpath)



    assert image_filename == mesh_depth_filename
    assert image_filename == validity_map_filename
    assert image_filename == ground_truth_filename

    if not paths_only:
        cv2.imwrite(image_outpath, imagec)

    return (image_refpath,
            image_outpath,
            mesh_depth_outpath,
            validity_map_outpath,
            ground_truth_outpath)


parser = argparse.ArgumentParser()

parser.add_argument('--paths_only', action='store_true')

args = parser.parse_args()


data_dirpaths = [
    NYU_DATA_150_DIRPATH
]

train_output_filepaths = [
    [
        NYU_TRAIN_IMAGE_150_FILEPATH,
        NYU_TRAIN_MESH_DEPTH_150_FILEPATH,
        NYU_TRAIN_VALIDITY_MAP_150_FILEPATH,
        NYU_TRAIN_GROUND_TRUTH_150_FILEPATH,
        NYU_TRAIN_INTRINSICS_150_FILEPATH
    ]
]
test_output_filepaths = [
    [
        NYU_TEST_IMAGE_150_FILEPATH,
        NYU_TEST_MESH_DEPTH_150_FILEPATH,
        NYU_TEST_VALIDITY_MAP_150_FILEPATH,
        NYU_TEST_GROUND_TRUTH_150_FILEPATH,
        NYU_TEST_INTRINSICS_150_FILEPATH
    ]
]
unused_output_filepaths = [
    [
        NYU_UNUSED_IMAGE_150_FILEPATH,
        NYU_UNUSED_MESH_DEPTH_150_FILEPATH,
        NYU_UNUSED_VALIDITY_MAP_150_FILEPATH,
        NYU_UNUSED_GROUND_TRUTH_150_FILEPATH,
        NYU_UNUSED_INTRINSICS_150_FILEPATH
    ]
]



for dirpath in [TRAIN_REFS_DIRPATH, TEST_REFS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

data_filepaths = \
    zip(data_dirpaths, train_output_filepaths, test_output_filepaths, unused_output_filepaths)

for data_dirpath, train_filepaths, test_filepaths, unused_filepaths in data_filepaths:
    # Training set
    train_image_filepath = os.path.join(data_dirpath, NYU_TRAIN_IMAGE_FILENAME)
    train_mesh_depth_filepath = os.path.join(data_dirpath, NYU_TRAIN_MESH_DEPTH_FILENAME)
    train_validity_map_filepath = os.path.join(data_dirpath, NYU_TRAIN_VALIDITY_MAP_FILENAME)
    train_ground_truth_filepath = os.path.join(data_dirpath, NYU_TRAIN_GROUND_TRUTH_FILENAME)
    train_intrinsics_filepath = os.path.join(data_dirpath, NYU_TRAIN_INTRINSICS_FILENAME)

    # Read training paths
    train_image_paths = data_utils.read_paths(train_image_filepath)
    train_mesh_depth_paths = data_utils.read_paths(train_mesh_depth_filepath)
    train_validity_map_paths = data_utils.read_paths(train_validity_map_filepath)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_filepath)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_filepath)

    
    print(len(train_image_paths), len(train_mesh_depth_paths))
    assert len(train_image_paths) == len(train_mesh_depth_paths)
    assert len(train_image_paths) == len(train_validity_map_paths)
    assert len(train_image_paths) == len(train_ground_truth_paths)
    assert len(train_image_paths) == len(train_intrinsics_paths)

    # Testing set
    test_image_filepath = os.path.join(data_dirpath, NYU_TEST_IMAGE_FILENAME)
    test_mesh_depth_filepath = os.path.join(data_dirpath, NYU_TEST_MESH_DEPTH_FILENAME)
    test_validity_map_filepath = os.path.join(data_dirpath, NYU_TEST_VALIDITY_MAP_FILENAME)
    test_ground_truth_filepath = os.path.join(data_dirpath, NYU_TEST_GROUND_TRUTH_FILENAME)
    test_intrinsics_filepath = os.path.join(data_dirpath, NYU_TEST_INTRINSICS_FILENAME)

    # Read testing paths
    test_image_paths = data_utils.read_paths(test_image_filepath)
    test_mesh_depth_paths = data_utils.read_paths(test_mesh_depth_filepath)
    test_validity_map_paths = data_utils.read_paths(test_validity_map_filepath)
    test_ground_truth_paths = data_utils.read_paths(test_ground_truth_filepath)
    test_intrinsics_paths = data_utils.read_paths(test_intrinsics_filepath)

    assert len(test_image_paths) == len(test_mesh_depth_paths)
    assert len(test_image_paths) == len(test_validity_map_paths)
    assert len(test_image_paths) == len(test_ground_truth_paths)
    assert len(test_image_paths) == len(test_intrinsics_paths)

    # Get test set directories
    test_seq_dirpaths = set(
        [test_image_paths[idx].split(os.sep)[-3] for idx in range(len(test_image_paths))])

    # Initialize placeholders for training output paths
    train_image_outpaths = []
    train_mesh_depth_outpaths = []
    train_validity_map_outpaths = []
    train_ground_truth_outpaths = []
    train_intrinsics_outpaths = []

    # Initialize placeholders for testing output paths
    test_image_outpaths = []
    test_mesh_depth_outpaths = []
    test_validity_map_outpaths = []
    test_ground_truth_outpaths = []
    test_intrinsics_outpaths = []

    # Initialize placeholders for unused testing output paths
    unused_image_outpaths = []
    unused_mesh_depth_outpaths = []
    unused_validity_map_outpaths = []
    unused_ground_truth_outpaths = []
    unused_intrinsics_outpaths = []

    # For each dataset density, grab the sequences
    seq_dirpaths = glob.glob(os.path.join(data_dirpath, 'data', '*'))
    n_sample = 0

    for seq_dirpath in seq_dirpaths:

        print('-'*10)
        print('cur: [%s]' %(seq_dirpath.split('/')[-1]))
        
        # For each sequence, grab the images, sparse depths and valid maps
        image_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'image', '*.png')))
        mesh_depth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'mesh_depth', '*.png')))
        validity_map_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'validity_map', '*.png')))
        ground_truth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'ground_truth', '*.png')))
        intrinsics_path = os.path.join(seq_dirpath, 'K.txt')

        print('image: %d  mesh: %d  validity: %d  unused: %d' %(len(image_paths), len(mesh_depth_paths), len(validity_map_paths), unused_img))
        assert len(image_paths) == len(mesh_depth_paths)+unused_img
        assert len(image_paths) == len(validity_map_paths)+unused_img

        # Load intrinsics
        kin = np.loadtxt(intrinsics_path)

        intrinsics_refpath = \
            os.path.join(*intrinsics_path.split(os.sep)[2:])
        intrinsics_outpath = \
            os.path.join(NYU_OUTPUT_DIRPATH, intrinsics_refpath[:-3]+'npy')
        image_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'image')

        if not os.path.exists(image_out_dirpath):
            os.makedirs(image_out_dirpath)

        # Save intrinsics
        np.save(intrinsics_outpath, kin)

        # do not skip first images
        if (seq_dirpath.split(os.sep)[-1] in test_seq_dirpaths):
            start_idx = 0
            offset_idx = 0
        else:
            start_idx = 5
            offset_idx = 5

        pool_input = []
        for idx in range(start_idx, len(image_paths)-offset_idx-start_idx):
            pool_input.append((
                image_paths[idx-offset_idx],
                image_paths[idx],
                image_paths[idx+offset_idx],
                mesh_depth_paths[idx],
                validity_map_paths[idx],
                ground_truth_paths[idx],
                args.paths_only))

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)

            for result in pool_results:
                image_refpath, \
                    image_outpath, \
                    mesh_depth_outpath, \
                    validity_map_outpath, \
                    ground_truth_outpath = result

                # Split into training, testing and unused testing sets
                #print('\n')
                image_refpath = os.path.join(dataset_name, image_refpath)
                #print(image_refpath)
                #print(len(train_image_paths))
                #assert 0
                if image_refpath in train_image_paths:
                    #print('append to TRAIN')
                    train_image_outpaths.append(image_outpath)
                    train_mesh_depth_outpaths.append(mesh_depth_outpath)
                    train_validity_map_outpaths.append(validity_map_outpath)
                    train_ground_truth_outpaths.append(ground_truth_outpath)
                    train_intrinsics_outpaths.append(intrinsics_outpath)
                elif image_refpath in test_image_paths:
                    #print('append to TEST')
                    test_image_outpaths.append(image_outpath)
                    test_mesh_depth_outpaths.append(mesh_depth_outpath)
                    test_validity_map_outpaths.append(validity_map_outpath)
                    test_ground_truth_outpaths.append(ground_truth_outpath)
                    test_intrinsics_outpaths.append(intrinsics_outpath)
                else:
                    #print('append to UNSUED')
                    unused_image_outpaths.append(image_outpath)
                    unused_mesh_depth_outpaths.append(mesh_depth_outpath)
                    unused_validity_map_outpaths.append(validity_map_outpath)
                    unused_ground_truth_outpaths.append(ground_truth_outpath)
                    unused_intrinsics_outpaths.append(intrinsics_outpath)

        n_sample = n_sample + len(pool_input)

        print('Completed processing {} examples for sequence={}'.format(
            len(pool_input), seq_dirpath))

    print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))

    nyu_train_image_filepath, \
        nyu_train_mesh_depth_filepath, \
        nyu_train_validity_map_filepath, \
        nyu_train_ground_truth_filepath, \
        nyu_train_intrinsics_filepath = train_filepaths

    print('Storing {} training image file paths into: {}'.format(
        len(train_image_outpaths), nyu_train_image_filepath))
    data_utils.write_paths(
        nyu_train_image_filepath, train_image_outpaths)

    print('Storing {} training sparse depth file paths into: {}'.format(
        len(train_mesh_depth_outpaths), nyu_train_mesh_depth_filepath))
    data_utils.write_paths(
        nyu_train_mesh_depth_filepath, train_mesh_depth_outpaths, USE_MESH)

    print('Storing {} training validity map file paths into: {}'.format(
        len(train_validity_map_outpaths), nyu_train_validity_map_filepath))
    data_utils.write_paths(
        nyu_train_validity_map_filepath, train_validity_map_outpaths)

    print('Storing {} training groundtruth depth file paths into: {}'.format(
        len(train_ground_truth_outpaths), nyu_train_ground_truth_filepath))
    data_utils.write_paths(
        nyu_train_ground_truth_filepath, train_ground_truth_outpaths)

    print('Storing {} training camera intrinsics file paths into: {}'.format(
        len(train_intrinsics_outpaths), nyu_train_intrinsics_filepath))
    data_utils.write_paths(
        nyu_train_intrinsics_filepath, train_intrinsics_outpaths)

    nyu_test_image_filepath, \
        nyu_test_mesh_depth_filepath, \
        nyu_test_validity_map_filepath, \
        nyu_test_ground_truth_filepath, \
        nyu_test_intrinsics_filepath = test_filepaths

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_outpaths), nyu_test_image_filepath))
    data_utils.write_paths(
        nyu_test_image_filepath, test_image_outpaths)

    print('Storing {} testing sparse depth file paths into: {}'.format(
        len(test_mesh_depth_outpaths), nyu_test_mesh_depth_filepath))
    data_utils.write_paths(
        nyu_test_mesh_depth_filepath, test_mesh_depth_outpaths, USE_MESH)

    print('Storing {} testing validity map file paths into: {}'.format(
        len(test_validity_map_outpaths), nyu_test_validity_map_filepath))
    data_utils.write_paths(
        nyu_test_validity_map_filepath, test_validity_map_outpaths)

    print('Storing {} testing groundtruth depth file paths into: {}'.format(
        len(test_ground_truth_outpaths), nyu_test_ground_truth_filepath))
    data_utils.write_paths(
        nyu_test_ground_truth_filepath, test_ground_truth_outpaths)

    print('Storing {} testing camera intrinsics file paths into: {}'.format(
        len(test_intrinsics_outpaths), nyu_test_intrinsics_filepath))
    data_utils.write_paths(
        nyu_test_intrinsics_filepath, test_intrinsics_outpaths)

    nyu_unused_image_filepath, \
        nyu_unused_mesh_depth_filepath, \
        nyu_unused_validity_map_filepath, \
        nyu_unused_ground_truth_filepath, \
        nyu_unused_intrinsics_filepath = unused_filepaths

    print('Storing {} unused testing image file paths into: {}'.format(
        len(unused_image_outpaths), nyu_unused_image_filepath))
    data_utils.write_paths(
        nyu_unused_image_filepath, unused_image_outpaths)

    print('Storing {} unused testing sparse depth file paths into: {}'.format(
        len(unused_mesh_depth_outpaths), nyu_unused_mesh_depth_filepath))
    data_utils.write_paths(
        nyu_unused_mesh_depth_filepath, unused_mesh_depth_outpaths, USE_MESH)

    print('Storing {} unused testing validity map file paths into: {}'.format(
        len(unused_validity_map_outpaths), nyu_unused_validity_map_filepath))
    data_utils.write_paths(
        nyu_unused_validity_map_filepath, unused_validity_map_outpaths)

    print('Storing {} unused testing groundtruth depth file paths into: {}'.format(
        len(unused_ground_truth_outpaths), nyu_unused_ground_truth_filepath))
    data_utils.write_paths(
        nyu_unused_ground_truth_filepath, unused_ground_truth_outpaths)

    print('Storing {} unused testing camera intrinsics file paths into: {}'.format(
        len(unused_intrinsics_outpaths), nyu_unused_intrinsics_filepath))
    data_utils.write_paths(
        nyu_unused_intrinsics_filepath, unused_intrinsics_outpaths)
