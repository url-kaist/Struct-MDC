import imageio
import argparse
import numpy as np
import os 

#####################################################################
# remove items if number of detected features are less then < threshold, 
# or there doesn't exist mesh
#####################################################################





###### Configurations ######
dirs = ['ground_truth', 'visualize', 
        'image', 
        'sparse_depth', 'validity_map', 
        'mesh_uv'
        ]

DEBUG = False
POINT_ALSO = False
###### Configurations ######


def remove_feat(args):

    path = os.path.join(args.root_path, args.dataset_name)
    path = os.path.join(path, 'data')

    # get test seqs.
    train_seqs = []
    f = open(path.replace('data', 'sequences.txt'), 'r')
    while True:
        line = f.readline()
        if not line:
            break

        s = line[:-1]
        train_seqs.append(s)
    f.close()

    remove = []
    max_d = 0

    avg_density = 0
    density = 0
    total = 0

                                                                                                                                                                                            
    for seq in train_seqs:
        seq_name = path+'/'+seq

        # remove # of feature < thr
        for a,b,files in os.walk(seq_name+'/'+'validity_map'):

            for f in files:
                f_path = a+'/'+f
                img = imageio.imread(f_path)
                num_feat = np.count_nonzero(img)

                ##### DEBUG INFO #####
                if DEBUG:
                    W,H = img.shape
                    total += 1

                    if num_feat >= 460:
                        density += 1
                    avg_density += 100*(num_feat) / (H*W)
                    print('avg density: %f      [removal is skipped!]' %(avg_density/total) )
                    continue
                ##### DEBUG INFO #####


                # check mesh .csv
                csv_path = f_path.replace('validity_map', 'mesh_uv').replace('png','csv')
                csv = np.genfromtxt(csv_path,delimiter=',')

                if num_feat < args.min_feat or csv.ndim < 2:
                    remove.append(f)

                    for d in dirs:
                        tmp_path = f_path.replace('validity_map', d)
                        if d == 'mesh_uv':
                            tmp_path = tmp_path.replace('png', 'csv')
                        elif d == 'pose' or d == 'pose_uncertainty':
                            tmp_path = tmp_path.replace('png', 'txt')

                        if os.path.isfile(tmp_path):
                            os.remove(tmp_path)
                            if POINT_ALSO:
                                if os.path.isfile(tmp_path.replace('PLAD_v2', 'PLAD_point_v2')):
                                    os.remove(tmp_path.replace('PLAD_v2', 'PLAD_point_v2'))

                    print('removed: %s' %(f_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")

    parser.add_argument("--min_feat",      type=int, default=150, help="minium number of features (threshold)")

    args = parser.parse_args()

    remove_feat(args)
