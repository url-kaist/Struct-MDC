import os
import argparse
import sys
from typing import DefaultDict




#####################################################################
# remove the interval you specified. It is used for cut-off 
# Based on the frame index [i] you defined, 
# i)  mode 1 : cut_start = True (all previous frames)
#       remove [0] ~ [i-1]
#
# ii) mode 2 : cut_start = False (all subsequent frames)
#       remove [i] ~ [end]
#
# This allows you to split a sequence into two sequences.
#####################################################################




###### Configurations ######
dirs = ['ground_truth', 'visualize', 
        'image', 
        'sparse_depth', 'validity_map', 
        'mesh_uv'
        ]


REMOVE_POINT = False

# [idx] lives at end split 
seq = 'MH_04_difficult' # sequence name
idx = 1403637259.388319 # index name (= timestamp in file_name), excluding format string, like '.png'
cut_start = False # whether remove [start part | end part]
###### Configurations ######






def remove_interval(args):

    path = os.path.join(args.root_path, args.dataset_name)
    data_path = os.path.join(path, 'data')

    # remove first ~ [remove_idx-1]
    if cut_start:
        seq_path = os.path.join(data_path, seq)

        # get all remove idices
        remove_idx = []
        for _, _, files in os.walk(os.path.join(seq_path, 'validity_map')):
            files.sort()

            for f in files:
                if f <= str(idx):
                    remove_idx.append(f[:-4])
    # remove [remove_idx] ~ end
    else:
        seq_path = os.path.join(data_path, seq)

        # get all remove idices
        remove_idx = []
        for _, _, files in os.walk(os.path.join(seq_path, 'validity_map')):
            files.sort()

            for f in files:
                if f >= str(idx):
                    remove_idx.append(f[:-4])


    # do remove
    for d in dirs:
        for rid in remove_idx:
            rpath = os.path.join(os.path.join(seq_path, d), rid)

            if os.path.exists(rpath+'.png'):
                os.remove(rpath+'.png')
                if REMOVE_POINT:
                    os.remove(rpath.replace(args.dataset_name, args.subdataset_name)+'.png')
            elif os.path.exists(rpath+'.csv'):
                os.remove(rpath+'.csv')
            elif os.path.exists(rpath+'.txt'):
                os.remove(rpath+'.txt')
                if REMOVE_POINT:
                    os.remove(rpath.replace(args.dataset_name, args.subdataset_name)+'.txt')

    print('Removed %d in %s' %(len(remove_idx), seq))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")
    parser.add_argument("--subdataset_name",  type=str, default="my_data_sub", help="not used")


    args = parser.parse_args()

    remove_interval(args)
