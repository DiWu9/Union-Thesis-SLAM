import numpy as np

from data_structures import hash_entry as he
from data_structures import voxel as v


class Bucket:

    def __init__(self, bucket_size=5):
        self._bucket_size = bucket_size
        self._bucket_list = [None] * self._bucket_size

    def get_ith_entry(self, i):
        try:
            return self._bucket_list[i]
        except IndexError:
            print("bucket.get_ith_entry: index out of range")

    def set_ith_entry(self, hash_entry, i):
        self._bucket_list[i] = hash_entry

    def remove_ith_entry(self, i):
        self._bucket_list[i] = None

    def is_full(self):
        """
        check if the bucket is full or not (full doesn't mean overflow)
        if there's one element in the bucket is None, it means the bucket is not full
        """
        for element in self._bucket_list:
            if element is None:
                return False
        return True

    def is_overflow(self):
        """
        when the bucket if full, check the last element's offset to check overflow
        """
        if self.is_full():
            for hash_entry in self._bucket_list:
                if not hash_entry.is_empty_offset():
                    return True
            return False
        else:
            return False

    def is_empty(self):
        """
        check if the bucket is empty
        cannot check it by len(bucket) == 0 because [None] has length 1
        """
        return np.array_equal(self._bucket_list, [None] * self._bucket_size)

    def get_num_entry_stored(self):
        """
        check how many entries are stored in the bucket
        """
        count = 0
        for hash_entry in self._bucket_list:
            if hash_entry is not None:
                count += 1
        return count

    def add_hash_entry(self, hash_entry):
        """
        add the hash entry to the bucket
        :return: the index where the hash_entry is stored if success, else -1
        """
        for i in range(self._bucket_size):
            if self._bucket_list[i] is None:
                self._bucket_list[i] = hash_entry
                return i
        return -1

    def get_hash_entry(self, world_coord):
        """
        :return: None if the hash entry is not found in the bucket
        """
        temp_entry = he.HashEntry(world_coord, None, None)
        if self.contains(temp_entry):
            for bucket_entry in self._bucket_list:
                if bucket_entry.equals(temp_entry):
                    return bucket_entry
        return None

    def remove_hash_entry(self, world_coord):
        """
        remove the hash entry of specified world coordinate
        :return: index if removed successfully, else -1
        """
        temp_entry = he.HashEntry(world_coord, None, None)
        for i in range(self._bucket_size):
            element = self._bucket_list[i]
            if element is None:
                continue
            if temp_entry.equals(element):
                self._bucket_list[i] = None
                return i
        return -1

    def contains(self, hash_entry):
        """
        check if the bucket contains the hash entry
        """
        if self.is_empty():
            return False
        else:
            for element in self._bucket_list:
                if element is None:
                    continue
                if element.equals(hash_entry):
                    return True
            return False


if __name__ == '__main__':
    for i in range(5):
        print(i)
    print(i)
