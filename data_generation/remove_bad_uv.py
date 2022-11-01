######################################
#        Data generation tool        #
######################################

# AUTHOR LICENSE INFO.
######################################
'''
Author: Jinwoo Jeon <zinuok@kaist.ac.kr>
If you use this code, please cite the following paper:
J. Jeon, H. Lim, D. Seo, H. Myung. "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM"
'''

# DESCRIPTIONS
######################################
'''
  - remove items if there doesn't exist mesh
'''




import argparse
import numpy as np
import os 





###### Configurations ######
dirs = ['ground_truth', 'visualize', 
        'image', 
        'sparse_depth', 'validity_map', 
        'mesh_uv'
        ]
###### Configurations ######

def remove_uv(args):

    path = os.path.join(args.root_path, args.dataset_name)
    path = os.path.join(path, 'data')

    # get test seqs.
    train_seqs = []
    f = open(path.replace('data', 'sequences.txt'), 'r')
    while True:
        line = f.readline()
        if not line:
            break

        s = line.strip('\n') #[:-1]
        train_seqs.append(s)
    f.close()



    for seq in train_seqs:
        seq_name = os.path.join(path, seq)

        # remove # of feature < 100
        for a,b,val_files in os.walk(os.path.join(seq_name, 'validity_map')):
            pass
        for a,b,uv_files in os.walk(os.path.join(seq_name, 'mesh_uv')):
            pass


        remove = []
        if len(val_files) != len(uv_files):
            for uv in uv_files:
                uv_ = uv.replace('csv', 'png')
                if uv_ not in val_files:
                    remove.append(uv)
            for r in remove:
                tmp_path_ = os.path.join(a, r) 
                os.remove(tmp_path_)

            print(len(val_files), len(uv_files), len(uv_files)-len(val_files), len(remove))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")


    args = parser.parse_args()

    remove_uv(args)
