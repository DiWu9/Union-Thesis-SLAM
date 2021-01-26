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

        self._table_size = 1000000
        self._hash_table = [None] * self._table_size
        self._num_entries_stored = 0
        print("Number of buckets: {}".format(self._table_size))
        self._bucket_size = 5

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
            return np.remainder(np.bitwise_xor(np.bitwise_xor(x * p1, y * p2), z * p3), self._table_size)
        except IndexError:
            print("hash_fusion.hash_function: invalid world_coord input.")

    def add(self, voxel, world_coord):
        """
        add the (world_coord, voxel) pair to the hash map
        """
        hash_entry = he.HashEntry(world_coord, None, voxel)
        hash_value = self.hash_function(world_coord)
        bucket = self.get_ith_bucket(hash_value)

    def add_hash_entry(self, hash_entry):
        """
        add the hash_entry to the hash map
        resolve collisions
        """

    def get_voxel(self, world_coord):
        """
        get voxel by its world coordinate
        """

    def get_hash_entry(self, world_coord):
        """
        get hash entry by its world coordinate
        """

    def remove(self, world_coord):
        """
        remove the voxel in the given world coordinate
        if there exist a voxel, return the voxel; else return None
        """

    def get_ith_bucket(self, i):
        try:
            return self._hash_table[i]
        except IndexError:
            print("hash_fusion.get_ith_bucket: invalid index")

    def garbage_collection(self):
        """after the image integration, remove voxels that have 0 weight"""

    def _add_entry_num(self):
        self._num_entries_stored += 1

    def _estimate_hash_table_size_by_voxel_grid_dimension(self, load_factor=0.75):
        """
        estimate the number of buckets needed by the hash table
        """
        # 10 is a magic number and it needs to be tested
        num_points_estimate = self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2] / 10
        return num_points_estimate / load_factor

    def _double_table_size(self):
        """
        resize the hash table to accommodate more hash entries within load factor of 0.75
        """
        print("Resizing hash table from {} to {}".format(self._table_size, self._table_size * 2))

        # extracting all hash entries to the temporary entry buffer
        temp_entry_buffer = []
        for i in range(self._table_size):
            entries_of_hash_value_i = self._get_hash_entries_by_hash_value(i)
            temp_entry_buffer = temp_entry_buffer + entries_of_hash_value_i
        print("Current number of hash entries: {}.".format(len(temp_entry_buffer)))

        # create the new hash table and fill all hash entries in
        self._table_size = self._table_size * 2
        self._hash_table = [None] * self._table_size
        for hash_entry in temp_entry_buffer:
            self.add_hash_entry(hash_entry)

        print("Resize finished.")

    def _get_hash_entries_by_hash_value(self, hash_value):
        """
        get all hash entries of the specified hash values
        """
        entry_list = []
        bucket = self.get_ith_bucket(hash_value)

        # returns empty list if bucket does not exist
        if bucket is None:
            return entry_list

        # iterate hash entries in the bucket
        last_entry = None
        for i in range(self._bucket_size):
            hash_entry = bucket.get_ith_entry(i)
            if hash_entry is None:
                continue
            else:
                if self._in_corresponding_bucket(hash_entry, hash_value):
                    entry_list.append(hash_entry)
                    last_entry = hash_entry

        # iterate linked list if exists
        if bucket.is_overflow() and not last_entry.is_empty_offset():
            pointer = last_entry.get_offset()
            while pointer is not None:
                ith_bucket = pointer[0]
                ith_entry = pointer[1]
                overflowed_entry = self.get_ith_bucket(ith_bucket).get_ith_entry(ith_entry)
                if self._in_corresponding_bucket(overflowed_entry, hash_value):
                    pointer = overflowed_entry.get_offset()
                    entry_list.append(overflowed_entry)

        return entry_list

    def _in_corresponding_bucket(self, hash_entry, bucket_index):
        """
        :return: True when hash entry's value is the index of the bucket, else False (it is pointed by an offset)
        """
        return bucket_index == self.hash_function(hash_entry.get_position())


if __name__ == '__main__':
    voxel_container = HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, False)
    world_coords = [[2, 5, 2], [4, 6, 2], [64, 2, 65], [23, 5, 63]]
    for world_coord in world_coords:
        print(voxel_container.hash_function(world_coord))
