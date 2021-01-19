import numpy as np

from data_structures import voxel as v


class HashEntry:

    def __init__(self, position, offset, voxel):
        """
        world world_coord
        offset pointer to handle collisions
        pointer to the voxel block
        """
        self.position = position
        self.offset = offset
        self.voxel = voxel


if __name__ == '__main__':
    print("Test hash entry")
    entry = HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
    print("Test finished")
