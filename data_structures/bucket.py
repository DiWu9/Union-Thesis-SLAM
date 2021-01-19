import numpy as np

from data_structures import hash_entry as he
from data_structures import voxel as v


class Bucket:

    def __init__(self, bucket_id, bucket_size=5):
        self._bucket_list = []
        self._bucket_size = bucket_size
        self._id = bucket_id  # not sure if useful

    def is_full(self):
        return len(self._bucket_list) >= self._bucket_size

    def is_empty(self):
        return len(self._bucket_list) <= 0

    def get_bucket_id(self):
        return self._id

    def get_num_entry_stored(self):
        return len(self._bucket_list)

    def get_bottom_entry(self):
        """
        when there bucket is full, get the bottom entry
        throw exception if bucket is not full
        """
        if self.is_full():
            return self._bucket_list[-1]
        else:
            raise Exception("bucket.get_bottom_entry: the bucket does not reach the bottom (not full)")

    def add_hash_entry(self, hash_entry):
        """
        add the hash entry to the bucket
        ## it assumes that the hash entry belongs to the bucket ##
        checking this needs hash function, hence it is done by the hash maps
        """
        if self.contains(hash_entry):
            raise Exception("bucket.add_hash_entry: entry collides with an existing one with same world coord")
        if self.is_full():
            if self._has_appended_list():
                appended_list = self.get_bottom_entry().get_appended_list()
                appended_list.append(hash_entry)
        else:
            self._bucket_list.append(hash_entry)

    def remove_hash_entry(self, world_coord):
        temp_entry = he.HashEntry(world_coord, None, None)
        if self.contains(temp_entry):
            return 0
            # implement remove

    def contains(self, hash_entry):
        """
        check if the bucket contains the hash entry
        """
        if self.is_empty():
            return False
        else:
            for bucket_entry in self._bucket_list:
                if bucket_entry.equals(hash_entry):
                    return True
            if self._has_appended_list():
                appended_list = self.get_bottom_entry().get_appended_list()
                for list_entry in appended_list:
                    if list_entry.equals(hash_entry):
                        return True
            return False

    def get_hash_entry(self, world_coord):
        temp_entry = he.HashEntry(world_coord, None, None)
        if self.contains(temp_entry):
            for bucket_entry in self._bucket_list:
                if bucket_entry.equals(temp_entry):
                    return bucket_entry
            if self._has_appended_list():
                appended_list = self.get_bottom_entry().get_appended_list()
                for list_entry in appended_list:
                    if list_entry.equals(temp_entry):
                        return list_entry
        return None

    def _has_appended_list(self):
        """
        when there is a bucket overflow, check if the bucket has already had a linked list
        """
        if self.is_full():
            last_entry = self.get_bottom_entry()
            return not last_entry.is_empty_offset()
        else:
            return False


if __name__ == '__main__':
    print('test bucket')
    entry1 = he.HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
    entry2 = he.HashEntry(np.array([1, 2, 5]), None, v.Voxel(None, None, None))
    entry3 = he.HashEntry(np.array([1, 2, 4]), None, v.Voxel(None, None, None))
    entry4 = he.HashEntry(np.array([1, 4, 3]), None, v.Voxel(None, None, None))
    entry5 = he.HashEntry(np.array([1, 7, 3]), None, v.Voxel(None, None, None))
    entry6 = he.HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
    entry_list = [entry1, entry2, entry3, entry4, entry5]
    bucket = Bucket(1)
    for entry in entry_list:
        bucket.add_hash_entry(entry)
    bucket.add_hash_entry(entry6)
    print('finish test')
