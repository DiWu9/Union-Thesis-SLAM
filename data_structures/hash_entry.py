import numpy as np

from data_structures import voxel as v


class HashEntry:

    def __init__(self, position, offset, voxel):
        """
        world world_coord
        offset pointer to handle collisions
        pointer to the voxel block
        """
        self._position = position
        self._offset = offset
        self._voxel = voxel

    def get_position(self):
        return self._position

    def is_empty_offset(self):
        return self._offset is None

    def get_appended_list(self):
        if self.is_empty_offset():
            return None
        else:
            return self._offset

    def equals(self, hash_entry):
        return np.array_equal(self._position, hash_entry.get_position())


if __name__ == '__main__':
    print("Test hash entry")
    entry = HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
    entry2 = HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
    print(entry.equals(entry2))
    print(np.array_equal(np.array([1, 2, 3]), np.array([1, 2, 3])))
    print("Test finished")
