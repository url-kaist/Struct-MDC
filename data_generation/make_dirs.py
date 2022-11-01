import os
import shutil
import argparse




dirs = ['ground_truth', 'pose', 
        'image', 
        'mesh_depth', 'mesh_uv',
        'sparse_depth', 'validity_map',
        'visualize']



def make_dirs(args):

    # make root dir
    path      = os.path.join(args.root_path, args.dataset_name)
    data_path = os.path.join(path, 'data')
    os.makedirs(data_path, exist_ok=True)

    # get sequence names
    f = open(args.sequence_list, 'r')
    seqs = []
    while True:
        line = f.readline()
        if not line:
            break
        seqs.append(line[:-1])
    f.close()

    # copy sequence list file into root dir
    shutil.copyfile(args.sequence_list, os.path.join(path, "sequences.txt"))

    # make a dir for each data sequence
    for seq in seqs:
        seq_path = os.path.join(data_path, seq)
        os.makedirs(seq_path, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(seq_path, d), exist_ok=True)
        
        # copy intrinsic K
        shutil.copyfile(args.intrinsics, os.path.join(seq_path, "K.txt"))

        print('[%s] has been generated.' % (seq))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")
    parser.add_argument("--sequence_list", type=str, default="/home/zinuok/Struct-MDC/data_generation/sequences.txt")
    parser.add_argument("--intrinsics",    type=str, default="/home/zinuok/Struct-MDC/data_generation/K.txt")

    args = parser.parse_args()

    make_dirs(args)
