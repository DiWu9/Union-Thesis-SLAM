import numpy as np
from numba import njit, prange
from skimage import measure

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0

class HashTable:
    """
    The data structure for voxel storage and retrieval: hash tables
    """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True):

    def integrate(self):
        """
        Integrate a RGB-D image to the hash table
        :return:
        """

    def generate_key(self, voxel):
        """
        generate the hash key of the given voxel and hash function:
        # H(x,y,z) = (x*p1 xor y*p2 xor z*p3) mod n
        :param voxel:
        :return: the hash key of the voxel
        """
        P_1 = 73856093
        P_2 = 19349669
        P_3 = 83492791
        return ()


class HashEntry:

    def __init__(self, position, offset, voxel):
        self.position = position
        self.offset = offset
        self.voxel = voxel

class Voxel:
    """

    """

    def __init__(self, dist, color, weight):
        self.dist = dist
        self.color = color
        self.weight = weight

    def integrate(self, new_dist, obs_weight=1.):
        """
        D’(v) = (D(v)W(v) + w_i(v)d_i(v)) / W(v) + w_i(v)
        W’(v) = W(v) + w_i(v)

        integrate the current image frame to the voxel
        :param new_dist: the distance of the voxel to the implicit surface
        :param obs_weight: the weight to assign for the current observation
        :return:
        """
        w_old = self.weight
        d_old = self.dist
        w_new = w_old + obs_weight
        d_new = (d_old * w_old + new_dist * obs_weight) / w_new
        self.weight = w_new
        self.dist = d_new

if __name__ == '__main__':
