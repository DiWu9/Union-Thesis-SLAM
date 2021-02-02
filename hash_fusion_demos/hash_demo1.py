"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import hash_fusion


if __name__ == "__main__":
    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    cam_intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))