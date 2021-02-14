import unittest
import numpy as np

from data_structures import bucket as b
from data_structures import hash_entry as he
from data_structures import voxel as v

entry1 = he.HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))
entry2 = he.HashEntry(np.array([2, 0, 5]), None, v.Voxel(None, None, None))
entry3 = he.HashEntry(np.array([3, 9, 4]), None, v.Voxel(None, None, None))
entry4 = he.HashEntry(np.array([4, 4, 3]), None, v.Voxel(None, None, None))
entry5 = he.HashEntry(np.array([5, 7, 3]), None, v.Voxel(None, None, None))
entry_list = [entry1, entry2, entry3, entry4, entry5]
entry6 = he.HashEntry(np.array([1, 2, 3]), None, v.Voxel(None, None, None))


class TestBucket(unittest.TestCase):

    def test_is_full(self):
        bucket = b.Bucket(5)
        self.assertFalse(bucket.is_full())
        for entry in entry_list:
            bucket.add_hash_entry(entry)
        self.assertTrue(bucket.is_full())

    def test_is_overflow(self):
        bucket = b.Bucket(5)
        self.assertFalse(bucket.is_overflow())
        for entry in entry_list:
            bucket.add_hash_entry(entry)
        self.assertEqual(bucket.get_num_entry_stored(), 5, "current number of entries is 5")
        self.assertFalse(bucket.is_overflow())
        entry = bucket.get_ith_entry(-1)
        entry.set_offset((2,3))
        self.assertTrue(bucket.is_overflow(), "last entry has non-empty offset = overflow")
        bucket.remove_hash_entry(np.array([4, 4, 3]))
        self.assertEqual(bucket.get_num_entry_stored(), 4, "current number of entries is 4")
        self.assertTrue(bucket.is_overflow(), "overflow can occur even if the bucket is not full")

    def test_add(self):
        bucket = b.Bucket(5)
        self.assertEqual(bucket.get_num_entry_stored(), 0, "test count method")
        for i in range(5):
            self.assertEqual(bucket.add_hash_entry(entry_list[i]), i, "return index where entry is added")
        self.assertEqual(bucket.add_hash_entry(entry6), -1, "return -1 when add failed")
        bucket.remove_hash_entry(np.array([1,2,3]))
        bucket.remove_hash_entry(np.array([3,9,4]))
        self.assertEqual(bucket.add_hash_entry(entry6), 0)

    def test_get(self):
        bucket = b.Bucket(5)
        for entry in entry_list:
            bucket.add_hash_entry(entry)
        self.assertTrue(entry2.equals(bucket.get_hash_entry(np.array([2,0,5]))))
        self.assertTrue(bucket.get_hash_entry(np.array([1,5,2])) is None)

    def test_remove(self):
        bucket = b.Bucket(5)
        for entry in entry_list:
            bucket.add_hash_entry(entry)
        self.assertEqual(bucket.remove_hash_entry(np.array([0,0,1])), -1)
        self.assertEqual(bucket.remove_hash_entry(np.array([3,9,4])), 2)

    def test_contains(self):
        bucket = b.Bucket(5)
        for entry in entry_list:
            bucket.add_hash_entry(entry)
        self.assertTrue(bucket.contains(entry6))
        entry7 = he.HashEntry(np.array([6, 7, 8]), None, v.Voxel(None, None, None))
        self.assertFalse(bucket.contains(entry7))


if __name__ == '__main__':
    unittest.main()