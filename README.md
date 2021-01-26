# Comparing the Performance of Volumetric Grids and Hash Tables for Mapping in SLAM

## Goal: 

Both volumetric grids and hash tables are used as the data structure for storing maps in SLAM. For my thesis research, I am trying to compare the running time and space efficiency of volumetric grids and hash tables in SLAM.
The example of volumetric grid is directly downloaded from [Volumetric TSDF Fusion of RGB-D Images in Python](https://github.com/andyzeng/tsdf-fusion-python). The implement of hash map is referenced from the [Voxel Hashing](https://github.com/niessner/VoxelHashing) method.

## Image Sets:

(Total Number) image sets are tested on both data structures for running time and space efficiency analysis.

1. A kitchen image sets with RGB-D image and each frame's camera pose are provided by the open source [Volumetric TSDF Fusion of RGB-D Images in Python](https://github.com/andyzeng/tsdf-fusion-python).
2.
3. 

## Progress:

#### 01/18/2021 
+ initial commit
+ finish constructor of bucket, voxel, hash entry, and hash map

#### 01/19/2021 
+ add data structures folder, which includes bucket, voxel, and hash entry
+ bucket: implement add, retrieval of hash entry (**Todo:** remove)
+ hash_fusion.py: implement hash table size estimation, hash function

#### 01/24/2021 
+ implement hash table resize

#### 01/26/2021 
+ modify bucket add, get, remove
+ finish bucket
+ add tests folder, add bucket tests
