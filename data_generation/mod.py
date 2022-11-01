import os
import argparse


objs = ["ground_truth", 
        "image", 
        "mesh_depth", 
        "validity_map", 
        "intrinsics"]


test_seqs = [
'test_1',
'test_2',
'test_3',
'test_4'
]



def gen_metadata(args):

    name = args.dataset_name
    path = os.path.join(args.root_path, name) + '/'

    ## 1) generate train meta data
    mode = 'train'
    print('[Training set]')
    for obj in objs:
        dest_obj = obj
        if obj == 'mesh_depth': 
            dest_obj = 'sparse_depth'
        dest_fname = path+mode+'_'+dest_obj+".txt"

        print('writing [%s]' %(dest_fname))
        outfile = open(dest_fname, "w")
        
        for _, sequences, _ in os.walk(path+"data/"):
            for seq_name in sequences:

                if seq_name in test_seqs:
                    continue
                
                if obj == "intrinsics":
                    obj_path = path+'data/'+seq_name+'/'+objs[0]+'/'
                    for r, s, files in os.walk(obj_path):

                        for f in files:
                            line = name+'/data/'+seq_name+'/K.txt\n'
                            outfile.write(line)
                else:
                    obj_path = path+'data/'+seq_name+'/'+obj+'/'
                    for r, s, files in os.walk(obj_path):

                        for f in files:
                            line = name+'/data/'+seq_name+'/'+obj+'/'+f+'\n'
                            outfile.write(line)

        outfile.close()



    ## 3) generate test meta data
    mode = 'test'
    print('\n[Test set]')
    for obj in objs:
        dest_obj = obj
        if obj == 'mesh_depth': 
            dest_obj = 'sparse_depth'
        dest_fname = path+mode+'_'+dest_obj+".txt"

        print('writing [%s]' %(dest_fname))
        outfile = open(dest_fname, "w")
        
        for _, sequences, _ in os.walk(path+"data/"):
            for seq_name in sequences:

                if seq_name not in test_seqs:
                    continue
                
                if obj == "intrinsics":
                    obj_path = path+'data/'+seq_name+'/'+objs[0]+'/'
                    for r, s, files in os.walk(obj_path):

                        for f in files:
                            line = name+'/data/'+seq_name+'/K.txt\n'
                            outfile.write(line)
                else:
                    obj_path = path+'data/'+seq_name+'/'+obj+'/'
                    for r, s, files in os.walk(obj_path):

                        for f in files:
                            line = name+'/data/'+seq_name+'/'+obj+'/'+f+'\n'
                            outfile.write(line)

        outfile.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")

    args = parser.parse_args()

    gen_metadata(args)
