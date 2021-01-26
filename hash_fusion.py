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

    def add_voxel(self, voxel, world_coord):
        """
        add the (world_coord, voxel) pair to the hash map
        """
        hash_entry = he.HashEntry(world_coord, None, voxel)
        self.add_hash_entry(hash_entry)

    def add_hash_entry(self, hash_entry):
        """
        add the hash_entry to the hash map
        resolve collisions
        :return: tuple (ith_bucket, ith_entry), which represents the place it's stored; 0 if add failed
        """
        if hash_entry is None:
            return 0
        hash_value = self.hash_function(hash_entry.get_position())
        bucket = self.get_ith_bucket(hash_value)
        if bucket is None:
            bucket = b.Bucket(self._bucket_size)
            bucket.add_hash_entry(hash_entry)
            self.set_ith_bucket(bucket, hash_value)
            self._num_entries_stored += 1
            return hash_value, 0
        else:
            if bucket.is_full():
                return self._add_to_linked_list(bucket, hash_value, hash_entry)
            else:
                return bucket.add_hash_entry(hash_entry)

    def _add_to_linked_list(self, corresponding_bucket, hash_value, hash_entry):
        """
        add the hash entry to the linked list
        """
        found_last_entry = False
        index = self._bucket_size - 1
        while not found_last_entry and index >= 0:
            hash_entry = corresponding_bucket.get_ith_entry(index)
            if self._in_corresponding_bucket(hash_entry, hash_value):
                found_last_entry = True
            else:
                index -= 1
        if found_last_entry:
            last_entry = hash_entry
            if last_entry.is_empty_offset():
                add_success = False
                bucket_index = hash_value + 1
                while not add_success:
                    bucket = self.get_ith_bucket(bucket_index)
                    if bucket.is_full():
                        # if the last bucket is full, scan the first bucket
                        if bucket_index >= self._table_size() - 1:
                            bucket_index = 0
                        else:
                            bucket_index += 1
                        continue
                    else:
                        ith_entry = bucket.add_hash_entry(hash_entry)
                        add_success = True
                        last_entry.set_offset((bucket_index, ith_entry))
                        self._num_entries_stored += 1
                        return bucket_index, ith_entry
        # if bucket is full and all of them belong to linked list (none of them "belongs" to this bucket)
        else:
            return  # implement

    def get_voxel(self, world_coord):
        """
        get voxel by its world coordinate
        """
        hash_entry = self.get_hash_entry(world_coords)
        return hash_entry.get_voxel()

    def get_hash_entry(self, world_coord):
        """
        get hash entry by its world coordinate
        """
        hash_value = self.hash_function(world_coord)
        temp_hash_entry = he.HashEntry(world_coord, None, None)
        entries_of_hash_value = self._get_hash_entries_by_hash_value(hash_value)
        for entry in entries_of_hash_value:
            if temp_hash_entry.equals(entry):
                return entry
        return None

    def _get_hash_entry(self, ith_bucket, ith_entry):
        """
        :return: None if bucket does not exist, else the entry
        """
        bucket = self.get_ith_bucket(ith_bucket)
        if bucket is not None:
            return bucket.get_ith_entry(ith_entry)
        else:
            return None

    def remove(self, world_coord):
        """
        remove the hash entry in the given world coordinate
        :return: 1 if remove done, 0 if remove failed
        """
        temp_hash_entry = he.HashEntry(world_coord, None, None)
        self.remove_hash_entry(temp_hash_entry)

    def remove_hash_entry(self, hash_entry):
        """
        remove the hash entry
        :return: 1 if remove done, 0 if remove failed
        """
        hash_value = self.hash_function(hash_entry.get_position())
        bucket = self.get_ith_bucket(hash_value)
        if bucket is None:
            return 0
        else:
            last_entry = None
            # case1: hash entry is in the corresponding bucket
            for i in range(self._bucket_size):
                element = bucket.get_ith_entry(i)
                if element is None:
                    continue
                else:
                    if self._in_corresponding_bucket(element, hash_value):
                        last_entry = element
                    if hash_entry.equals(element):
                        # 1a. if this hash entry is not in the linked list and has empty offset, simply remove it
                        if element.is_empty_offset():
                            bucket.remove_ith_entry(i)
                            return 1
                        # 1b. if this hash entry has an offset, replace this hash entry by the entry it points to
                        else:
                            pointer = hash_entry.get_offset()
                            next_element = self._get_hash_entry(pointer[0], pointer[1])
                            self._remove_hash_entry(pointer[0], pointer[1])
                            bucket.set_ith_entry(next_element)
                            return 1
            # case2: hash entry is in the linked list
            if last_entry is not None:
                current_entry = last_entry
                pointer = current_entry.get_offset()
                # iterate linked list
                while pointer is not None:
                    next_entry = self._get_hash_entry(pointer[0], pointer[1])
                    if hash_entry.equals(next_entry):
                        pointer_to_remove_entry = pointer
                        pointer = next_entry.get_offset()
                        # 2a. last element of list, remove it from hash table and remove current entry's offset
                        if pointer is None:
                            current_entry.set_offset(None)
                        # 2b. middle of list, remove it from table and set current entry's offset to next entry's offset
                        else:
                            current_entry.set_offset(next_entry.get_offset())
                        self._remove_hash_entry(pointer_to_remove_entry[0], pointer_to_remove_entry[1])
                        return 1
                    else:
                        current_entry = next_entry
                        pointer = current_entry.get_offset()
            return 0

    def _remove_hash_entry(self, ith_bucket, ith_entry):
        bucket = self.get_ith_bucket(ith_bucket)
        if bucket is not None:
            bucket.remove_ith_entry(ith_entry)

    def set_ith_bucket(self, bucket, i):
        self._hash_table[i] = bucket

    def get_ith_bucket(self, i):
        try:
            return self._hash_table[i]
        except IndexError:
            print("hash_fusion.get_ith_bucket: invalid index")

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
                overflowed_entry = self._get_hash_entry(ith_bucket, ith_entry)
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
