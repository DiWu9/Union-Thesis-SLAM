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

    def __init__(self, vol_bounds, voxel_size, map_size=1000000, use_gpu=True):
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

        self._table_size = map_size
        self._hash_table = [None] * self._table_size
        print("Number of buckets: {}".format(self._table_size))
        self._bucket_size = 5

    def integrate(self):
        """
        Integrate a RGB-D image to the hash table
        :return:
        """

    def count_num_hash_entries(self):
        num_hash_entries = 0
        for bucket in self._hash_table:
            if bucket is None:
                continue
            else:
                num_hash_entries += bucket.get_num_entry_stored()
        return num_hash_entries

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
            return -1, -1
        hash_value = self.hash_function(hash_entry.get_position())
        bucket = self.get_ith_bucket(hash_value)
        if bucket is None:
            bucket = b.Bucket(self._bucket_size)
            bucket.add_hash_entry(hash_entry)
            self.set_ith_bucket(bucket, hash_value)
            return hash_value, 0
        else:
            if bucket.is_full():
                return self._add_to_linked_list(bucket, hash_value, hash_entry)
            else:
                return hash_value, bucket.add_hash_entry(hash_entry)

    def _add_to_linked_list(self, corresponding_bucket, hash_value, hash_entry):
        """
        add the hash entry to the linked list
        """
        last_entry = corresponding_bucket.get_ith_entry(self._bucket_size - 1)
        if last_entry.is_empty_offset():
            return self._find_next_free_space_and_add_entry(hash_value, self._bucket_size - 1, last_entry, hash_entry)
        else:
            ith_bucket, ith_entry = last_entry.get_offset()
            next_on_chain = self._get_hash_entry(ith_bucket, ith_entry)
            while not next_on_chain.is_empty_offset():
                ith_bucket, ith_entry = next_on_chain.get_offset()
                next_on_chain = self._get_hash_entry(ith_bucket, ith_entry)
            return self._find_next_free_space_and_add_entry(ith_bucket, ith_entry, next_on_chain, hash_entry)

    def _find_next_free_space_and_add_entry(self, begin_b_idx, begin_e_idx, prev_entry, add_entry):
        """
        find the next available bucket to insert the hash entry
        :return: (idx_bucket, idx_entry) if inserted, else (-1,-1)
        """
        scanned_entire_map = False
        b_idx = begin_b_idx
        e_idx = begin_e_idx
        if e_idx < self._bucket_size - 1:
            bucket = self.get_ith_bucket(b_idx)
            # find next available space of current bucket to insert alien entry
            for i in range(e_idx, self._bucket_size - 1):
                if bucket.is_empty_entry(i):
                    bucket.set_ith_entry(add_entry, i)
                    prev_entry.set_offset((begin_b_idx, i))
                    return begin_b_idx, i
        # find the next available bucket to add entry
        while not scanned_entire_map:
            # set index to next bucket's index
            b_idx = 0 if b_idx >= self._table_size - 1 else b_idx + 1
            bucket = self.get_ith_bucket(b_idx)
            if bucket is None:
                bucket = b.Bucket(self._bucket_size)
                bucket.add_hash_entry(add_entry)
                self.set_ith_bucket(bucket, b_idx)
                prev_entry.set_offset((b_idx, 0))
                return b_idx, 0
            if bucket.is_full_for_alien_entries():
                scanned_entire_map = b_idx == begin_b_idx
            # bucket is neither none nor full
            else:
                ith_entry = bucket.add_hash_entry(add_entry)
                prev_entry.set_offset((b_idx, ith_entry))
                return b_idx, ith_entry
        return -1, -1

    def get_voxel(self, world_coord):
        """
        get voxel by its world coordinate
        """
        hash_entry = self.get_hash_entry(world_coord)
        return hash_entry.get_voxel()

    def get_hash_entry(self, world_coord):
        """
        get hash entry by its world coordinate
        """
        hash_value = self.hash_function(world_coord)
        temp_hash_entry = he.HashEntry(world_coord, None, None)
        bucket = self.get_ith_bucket(hash_value)
        if bucket is not None:
            if bucket.is_empty():
                return None
            last_entry = None
            # iterate bucket
            for i in range(self._bucket_size):
                entry = bucket.get_ith_entry(i)
                if temp_hash_entry.equals(entry):
                    return entry
                if entry is not None:
                    last_entry = entry
            # iterate linked list
            next_in_chain = last_entry
            while not next_in_chain.is_empty_offset():
                pointer = next_in_chain.get_offset()
                next_in_chain = self._get_hash_entry(pointer[0], pointer[1])
                if temp_hash_entry.equals(next_in_chain):
                    return next_in_chain
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
        if bucket.is_empty():
            return 0
        else:
            last_entry = None
            # case1: hash entry is in the corresponding bucket
            for i in range(self._bucket_size):
                element = bucket.get_ith_entry(i)
                if element is None:
                    continue
                else:
                    last_entry = element
                    if hash_entry.equals(element):
                        # 1a. if this hash entry is not in the linked list and has empty offset, simply remove it
                        if element.is_empty_offset():
                            bucket.remove_ith_entry(i)
                            return 1
                        # 1b. if last entry and has offset, set last entry to next element and delete next element
                        # if in corresponding bucket and has offset, then must be the last entry
                        else:
                            pointer = element.get_offset()
                            next_element = self._get_hash_entry(pointer[0], pointer[1])
                            self._remove_hash_entry(pointer[0],pointer[1])
                            bucket.set_ith_entry(next_element, i)
                            return 1
            # case2: hash entry is in the linked list
            if last_entry is not None:
                current_on_chain = last_entry
                # iterate linked list
                while not current_on_chain.is_empty_offset():
                    pointer = current_on_chain.get_offset()
                    next_entry = self._get_hash_entry(pointer[0], pointer[1])
                    if next_entry.equals(hash_entry):
                        next_pointer = next_entry.get_offset()
                        # 2a. last element of list, remove it from hash table and remove current entry's offset
                        if next_pointer is None:
                            current_on_chain.set_offset(None)
                            self._remove_hash_entry(pointer[0], pointer[1])
                        # 2b. middle of list, delete it and correct pointer of previous entry
                        else:
                            current_on_chain.set_offset(next_pointer)
                            self._remove_hash_entry(pointer[0], pointer[1])
                        return 1
                    current_on_chain = next_entry
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

    def get_next_bucket(self, i):
        """
        :return: next bucket in the hash table. (the next bucket of the last bucket is the first bucket)
        """
        # check if i is the last bucket
        if i >= self._table_size - 1:
            return self._hash_table[0]
        else:
            return self._hash_table[i+1]

    def _estimate_hash_table_size_by_voxel_grid_dimension(self, load_factor=0.75):
        """
        estimate the number of buckets needed by the hash table
        """
        # 10 is a magic number and it needs to be tested
        num_points_estimate = self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2] / 10
        return num_points_estimate / load_factor

    def double_table_size(self):
        """
        resize the hash table to accommodate more hash entries within load factor of 0.75
        """
        print("Resizing hash table from {} to {}".format(self._table_size, self._table_size * 2))

        original_hash_table = self._hash_table

        # create the new hash table and fill all hash entries in
        self._table_size = self._table_size * 2
        self._hash_table = [None] * self._table_size
        for bucket in original_hash_table:
            if bucket is None:
                continue
            else:
                for i in range(self._bucket_size):
                    entry = bucket.get_ith_entry(i)
                    if entry is None:
                        continue
                    else:
                        self.add_hash_entry(entry)
        print("Resize finished.")

    def _in_corresponding_bucket(self, hash_entry, bucket_index):
        """
        :return: True when hash entry's value is the index of the bucket, else False (it is pointed by an offset)
        """
        return bucket_index == self.hash_function(hash_entry.get_position())


if __name__ == '__main__':
    hash_map = HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 100000, False)
    world_coords = []
    for i in range(400000):
        x = np.random.randint(50)
        y = np.random.randint(50)
        z = i
        world_coords.append([x, y, z])
        position = [x, y, z]
        hash_entry = he.HashEntry(position, None, None)
        hash_value = hash_map.hash_function(position)
        bucket_index, entry_index = hash_map.add_hash_entry(hash_entry)
        print("Point {} of hash value {} is added to ({},{})".format(position, hash_value, bucket_index, entry_index))
    print(hash_map.count_num_hash_entries())
    print("Test add finished")

    for position in world_coords:
        entry = he.HashEntry(position, None, None)
        hash_map.remove_hash_entry(entry)
    print(hash_map.count_num_hash_entries())
    print("Test remove finished")