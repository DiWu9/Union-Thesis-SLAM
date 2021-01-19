import numpy as np

from data_structures import bucket as b
from data_structures import voxel as v
from data_structures import hash_entry as he

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

    def __init__(self, vol_bounds, voxel_size, use_gpu=True):
        print("Initializing hash maps ... ")
        vol_bounds = np.asarray(vol_bounds)
        assert vol_bounds.shape == (3, 2), "[!] `vol_bounds` should be of shape (3, 2)."

        self._vol_bounds = vol_bounds
        self._voxel_size = voxel_size
        self._trunc_margin = 5 * self._voxel_size
        self._color_const = 256 * 256
        self._vol_dim = np.ceil((self._vol_bounds[:, 1] - self._vol_bounds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bounds[:, 1] = self._vol_bounds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bounds[:, 0].copy(order='C').astype(np.float32)
        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2])
        )

        self._hash_table = []
        self._table_size = self.estimate_table_size_according_to_voxel_grid_dimension()
        self._bucket_size = 5
        for i in range(self._table_size):
            bucket = b.Bucket(self._bucket_size, i)
            self._hash_table.append(bucket)

    def estimate_table_size_according_to_voxel_grid_dimension(self):
        return self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2]

    def integrate(self):
        """
        Integrate a RGB-D image to the hash table
        :return:
        """

    def hash_function(self, world_coord):
        """
        generate the hash key of the given voxel and hash function:
            H(x,y,z) = (x*p1 xor y*p2 xor z*p3) mod n
        where n is the hash table size - the number of buckets
        :param world_coord: world coordinate (x,y,z) in np array
        :return: the hash key of the voxel
        """
        p1, p2, p3 = 73856093, 19349669, 83492791

        try:
            x, y, z = world_coord[0], world_coord[1], world_coord[2]
        except IndexError:
            print("hash_fusion.hash_function: invalid world_coord input.")

        return np.remainder(np.bitwise_xor(np.bitwise_xor(x * p1, y * p2), z * p3), self._table_size)

    def get_hash_entry(self, world_coord):
        """
        get the hash entry of the voxel
        """


if __name__ == '__main__':
    voxel_container = HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, False)
    world_coords = [[2, 5, 2], [4, 6, 2], [64, 2, 65], [23, 5, 63]]
    for world_coord in world_coords:
        print(voxel_container.hash_function(world_coord))
