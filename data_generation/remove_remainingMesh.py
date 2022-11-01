import imageio
import numpy as np
import os 


##############################
#   DEPRECATED !!            #
##############################

## remove items if number of features are less then < threshold, or 
## there doesn't exist mesh

test_seqs = [] 
# 'building_wall_1_1', 
# 'stairs_1_2'
# 'building_wall_3_2'
# 'piano_2', 
# 'rest_3_1', 

path = '/home/zinuok/Dataset/PLAD_v3/data'

# dirs = ['ground_truth', 'image', 'mesh_depth', 'mesh_uv', 'sparse_depth', 'validity_map', 'visualize']
dirs = ['ground_truth', 'image', 'sparse_depth', 'validity_map', 'visualize', 'pose']



# get test seqs.
train_seqs = []
f = open(path.replace('data', 'sequences.txt'), 'r')
while True:
    line = f.readline()
    if not line:
        break

    s = line.strip('\n') #[:-1]
    if s not in test_seqs:
        train_seqs.append(s)
f.close()


# train_seqs = ['corridor_2']

for seq in train_seqs:
    seq_name = path+'/'+seq

    # remove # of feature < 100
    for a,b,mesh_files in os.walk(seq_name+'/'+'mesh_depth'):
        pass
    for a_,b,img_files in os.walk(seq_name+'/'+'image'):
        pass


    remove = []
    if len(mesh_files) != len(img_files):
        for mesh in mesh_files:
            if mesh not in img_files:
                remove.append(mesh)
        for r in remove:
            tmp_path_ = a+'/'+r 
            os.remove(tmp_path_) 
            # os.remove(tmp_path_.replace('PLAD_new', 'PLAD_point_new'))

        print(len(mesh_files), len(img_files), len(mesh_files)-len(img_files), len(remove))
        
        # for f in files:
        #     f_path = a+'/'+f
        #     img = imageio.imread(f_path)
        #     num_feat = np.count_nonzero(img)

        #     # check mesh .csv
        #     csv_path = f_path.replace('validity_map', 'mesh_uv').replace('png','csv')
        #     csv = np.genfromtxt(csv_path,delimiter=',')

        #     if num_feat < 100 or csv.ndim < 2:
        #         remove.append(f)

        #         for d in dirs:
        #             tmp_path = f_path.replace('validity_map', d)
        #             if d == 'mesh_uv':
        #                 tmp_path = tmp_path.replace('png', 'csv')
        #             if os.path.isfile(tmp_path):
        #                 os.remove(tmp_path)
        #         print('removed: %s' %(f_path))

