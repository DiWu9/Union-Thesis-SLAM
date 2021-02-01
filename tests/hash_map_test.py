import unittest
import numpy as np

from data_structures import bucket as b
from data_structures import hash_entry as he
from data_structures import voxel as v
import hash_fusion as hf


class TestHashMap(unittest.TestCase):

    def test_resize_maintain_num_entries(self):
        hash_table = hf.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 10,
                                  False)
        world_coords = [[80, 56, 0], [66, 23, 1], [64, 5, 2], [77, 87, 3], [22, 55, 4], [1, 62, 5], [54, 98, 6],
                        [17, 35, 7], [42, 86, 8], [75, 84, 9], [72, 56, 10], [68, 94, 11], [31, 18, 12], [97, 83, 13],
                        [21, 56, 14], [16, 38, 15], [80, 46, 16], [22, 64, 17], [68, 79, 18], [98, 10, 19],
                        [26, 31, 20], [83, 53, 21], [11, 7, 22], [92, 7, 23], [76, 81, 24], [89, 75, 25], [2, 71, 26],
                        [82, 10, 27], [77, 58, 28], [57, 0, 29], [80, 25, 30], [43, 92, 31], [15, 26, 32], [33, 93, 33],
                        [23, 42, 34], [97, 22, 35], [6, 88, 36], [43, 51, 37], [6, 90, 38], [54, 83, 39], [18, 53, 40],
                        [7, 55, 41], [5, 64, 42], [15, 1, 43], [77, 25, 44], [82, 56, 45], [9, 44, 46], [54, 34, 47],
                        [0, 73, 48], [81, 95, 49]]
        for position in world_coords:
            hash_entry = he.HashEntry(position, None, None)
            hash_table.add_hash_entry(hash_entry)
        hash_table.double_table_size()
        self.assertEqual(50, hash_table.count_num_hash_entries())
        for position in world_coords:
            self.assertIsNotNone(hash_table.get_hash_entry(position))

    def test_add_until_full_size_10(self):
        hash_table = hf.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 10, False)
        world_coords = [[80, 56, 0], [66, 23, 1], [64, 5, 2], [77, 87, 3], [22, 55, 4], [1, 62, 5], [54, 98, 6],
                        [17, 35, 7], [42, 86, 8], [75, 84, 9], [72, 56, 10], [68, 94, 11], [31, 18, 12], [97, 83, 13],
                        [21, 56, 14], [16, 38, 15], [80, 46, 16], [22, 64, 17], [68, 79, 18], [98, 10, 19],
                        [26, 31, 20], [83, 53, 21], [11, 7, 22], [92, 7, 23], [76, 81, 24], [89, 75, 25], [2, 71, 26],
                        [82, 10, 27], [77, 58, 28], [57, 0, 29], [80, 25, 30], [43, 92, 31], [15, 26, 32], [33, 93, 33],
                        [23, 42, 34], [97, 22, 35], [6, 88, 36], [43, 51, 37], [6, 90, 38], [54, 83, 39], [18, 53, 40],
                        [7, 55, 41], [5, 64, 42], [15, 1, 43], [77, 25, 44], [82, 56, 45], [9, 44, 46], [54, 34, 47],
                        [0, 73, 48], [81, 95, 49]]
        for position in world_coords:
            hash_entry = he.HashEntry(position, None, None)
            hash_table.hash_function(position)
            hash_table.add_hash_entry(hash_entry)
        self.assertEqual(50, hash_table.count_num_hash_entries(), "50 entries are added")

    def test_add(self):
        hash_table = hf.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 10,
                                  False)
        world_coords = [[4, 43, 0], [48, 37, 1], [37, 33, 2], [0, 42, 3], [44, 11, 4], [2, 42, 5], [40, 0, 6],
                        [22, 43, 7], [22, 15, 8], [12, 4, 9], [38, 44, 10], [24, 8, 11], [18, 9, 12], [36, 26, 13],
                        [11, 42, 14], [5, 17, 15], [3, 38, 16], [13, 12, 17], [1, 43, 18], [18, 40, 19], [22, 3, 20],
                        [28, 40, 21], [3, 8, 22], [45, 25, 23], [15, 21, 24], [26, 49, 25], [9, 20, 26], [13, 42, 27],
                        [3, 47, 28], [3, 37, 29], [33, 14, 30], [46, 44, 31], [5, 22, 32], [33, 4, 33], [20, 41, 34],
                        [23, 48, 35], [35, 48, 36], [23, 25, 37], [9, 17, 38], [6, 38, 39], [25, 24, 40], [38, 47, 41],
                        [26, 20, 42], [31, 38, 43], [40, 42, 44], [23, 17, 45], [2, 23, 46], [6, 8, 47], [36, 39, 48],
                        [4, 45, 49]]
        for position in world_coords:
            hash_entry = he.HashEntry(position, None, None)
            hash_table.add_hash_entry(hash_entry)
        for i in range(10):
            bucket = hash_table.get_ith_bucket(i)
            for j in range(5):
                entry = bucket.get_ith_entry(j)
                position = entry.get_position()
                hash_val = hash_table.hash_function(position)
                print("({},{}) stores point {} of value {}".format(i, j, position, hash_val))
        entry = he.HashEntry([4, 43, 0], None, None)
        hash_table.remove_hash_entry(entry)
        entry_to_add = he.HashEntry([3,37,29], None, None)
        hash_table.add_hash_entry(entry_to_add)
        print("stop")

    def test_add_until_full_size_1000(self):
        hash_map = hf.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 1000, False)
        for i in range(5000):
            x = np.random.randint(500)
            y = np.random.randint(500)
            z = i
            position = [x, y, z]
            hash_entry = he.HashEntry(position, None, None)
            hash_value = hash_map.hash_function(position)
            bucket_index, entry_index = hash_map.add_hash_entry(hash_entry)
            print(
                "Point {} of hash value {} is added to ({},{})".format(position, hash_value, bucket_index, entry_index))
        self.assertEqual(5000, hash_map.count_num_hash_entries(), "5000 entries are added")

    def test_remove_all_entries_full_size_10(self):
        hash_table = hf.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02, 10,
                                  False)
        world_coords = [[4, 43, 0], [48, 37, 1], [37, 33, 2], [0, 42, 3], [44, 11, 4], [2, 42, 5], [40, 0, 6],
                        [22, 43, 7], [22, 15, 8], [12, 4, 9], [38, 44, 10], [24, 8, 11], [18, 9, 12], [36, 26, 13],
                        [11, 42, 14], [5, 17, 15], [3, 38, 16], [13, 12, 17], [1, 43, 18], [18, 40, 19], [22, 3, 20],
                        [28, 40, 21], [3, 8, 22], [45, 25, 23], [15, 21, 24], [26, 49, 25], [9, 20, 26], [13, 42, 27],
                        [3, 47, 28], [3, 37, 29], [33, 14, 30], [46, 44, 31], [5, 22, 32], [33, 4, 33], [20, 41, 34],
                        [23, 48, 35], [35, 48, 36], [23, 25, 37], [9, 17, 38], [6, 38, 39], [25, 24, 40], [38, 47, 41],
                        [26, 20, 42], [31, 38, 43], [40, 42, 44], [23, 17, 45], [2, 23, 46], [6, 8, 47], [36, 39, 48],
                        [4, 45, 49]]
        for position in world_coords:
            hash_entry = he.HashEntry(position, None, None)
            bidx, eidx = hash_table.add_hash_entry(hash_entry)
            print("point {} added to bucket {} entry {}".format(position, bidx, eidx))
        for position in world_coords:
            if position in [[2, 23, 46], [6, 8, 47], [25, 24, 40], [4, 45, 49]]:
                print("pause")
            hash_entry = he.HashEntry(position, None, None)
            hash_table.remove_hash_entry(hash_entry)
        print(hash_table.count_num_hash_entries())
        self.assertEqual(0, hash_table.count_num_hash_entries())



if __name__ == '__main__':
    unittest.main()