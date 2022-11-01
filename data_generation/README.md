
#### STEP 1) generate the same file structure as ours 

##### you have to prepare two files in advance: 'sequences.txt' and 'K.txt'
- sequences.txt: sequence list of your raw data [example]()
- K.txt: camera intrinsic parameters matrix [example]()

```bash
python3 data_generation/make_dirs.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME) \
        --sequence_list $(PATH_TO_SEQUENCE_LIST.TXT) \
        --dataset_name  $(PATH_TO_INTRINSICS.TXT) \
```


#### STEP 2) generate data from VIO (we use UV-SLAM)
- you have to build & source the your VIO package. In our case, UV-SLAM.
- if you use the UV-SLAM, modify the seq_path at [our code]()

```bash
roslaunch uv_slam plad.launch
```




#### STEP 3) Filter out inappropriate frames

##### Some frames have to be filtered out. For example, VIO initialization interval, bad feature, etc. We share the tools we used.

```bash 
# [filter] number of features less than threshold
python3 data_generation/remove_bad_feature.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME) \
        --min_feat      $(MINIMUM_THRESHOLD)

# [filter] inproper mesh
python3 data_generation/remove_bad_uv.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME) \
        --min_feat      $(MINIMUM_THRESHOLD)

# [filter] cut-off user defined interval (VIO initialization interval; [details]()) 
# after modifying lines 36-37 at remove_bad_gt.py (sequence name & frame index),
python3 data_generation/remove_bad_gt.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME)
```


#### STEP 4) generate [mesh_depth] using Triangulation. 
```bash
python3 data_generation/generate_sparse_depth.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME)
```


#### STEP 5) generate meta data for PyTorch dataloader
##### you have to prepare a file in advance: 'test_sequences.txt'
- test_sequences.txt: sequence list of your raw data for test/evaluation [example]()
```bash
python3 data_generation/mod.py \
        --root_path     $(YOUR_DATA_ROOT_PATH) \
        --dataset_name  $(YOUR_DATASET_NAME)
```



#### STEP 6) (Option) GT depth filling
- There are some missing holes in ground truth depth, due to sensor noise. 
- If you want, therefore, you can fill the holes using the following code. (modified from original [NYUv2 code]())
- file: data_generation/demo_fill_depth_colorization.m
  - the code is written on MATLAB
  - modify the path to your dataset where all sequence folders exist (line 14)


#### STEP 7) Parse the data for PyTorch dataset parser
```bash
cd Struct-MDC_src
mkdir data
ln -s $(PATH_TO_YOUR_DATASET_ROOT) data/
  
# please modify lines 27~35: 
#   - dataset name
#   - test sequence list
python3 setup/setup_dataset_plad.py
```
