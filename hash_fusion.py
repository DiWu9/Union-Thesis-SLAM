import numpy as np
import cProfile
from numba import njit, prange
from skimage import measure

import grid_fusion
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

    def __init__(self, vol_bounds, voxel_size, map_size=1000000, load_factor=0.75, use_gpu=False):
        print("Initializing hash maps ... ")
        # hash map attributes
        self._table_size = map_size
        self._hash_table = [None] * self._table_size
        self._load_factor = 0.75
        self._bucket_size = 5
        self._num_non_empty_bucket = 0

        # set up constants
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
        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing='ij'
        )
        self.vox_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """
        source: https://github.com/andyzeng/tsdf-fusion-python
        Convert voxel grid coordinates to world coordinates.
        """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """
        source: https://github.com/andyzeng/tsdf-fusion-python
        Convert camera coordinates to pixel coordinates.
        (which means to convert 3D coord to 2D coord)
        """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """
        reference: https://github.com/andyzeng/tsdf-fusion-python
        Integrate a RGB-D image to the hash table
        """
        im_h, im_w = depth_im.shape
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])
        cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
        cam_pts = grid_fusion.rigid_transform(cam_pts, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = self.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                    pix_z > 0))))
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.minimum(1, depth_diff / self._trunc_margin)
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        valid_pix_x = pix_x[valid_pts]
        valid_pix_y = pix_y[valid_pts]
        valid_dist = dist[valid_pts]

        # integration step
        for i in range(len(valid_vox_x)):
            # print(i)
            voxel_coord = [valid_vox_x[i], valid_vox_y[i], valid_vox_z[i]]
            target_entry = self.get_hash_entry(voxel_coord)
            if target_entry is None:
                voxel = v.Voxel()
                voxel.integrate(valid_dist[i], color_im[valid_pix_y[i], valid_pix_x[i]])
                entry = he.HashEntry(voxel_coord, None, voxel)
                self.add_hash_entry(entry)
            else:
                target_entry.integrate_voxel(valid_dist[i], color_im[valid_pix_y[i], valid_pix_x[i]])

    def count_num_hash_entries(self):
        num_hash_entries = 0
        for bucket in self._hash_table:
            if bucket is None:
                continue
            else:
                num_hash_entries += bucket.get_num_entry_stored()
        return num_hash_entries

    def needs_resize(self):
        """
        compute load factor and compare it with max load factor
        :returns: True if exceeds max load factor, else False
        """
        return self._num_non_empty_bucket / self._table_size >= self._load_factor

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
        entry_to_add = he.HashEntry(world_coord, None, voxel)
        self.add_hash_entry(entry_to_add)

    def add_hash_entry(self, hash_entry):
        """
        add the hash_entry to the hash map
        resolve collisions
        :return: tuple (ith_bucket, ith_entry), which represents the place it's stored; 0 if add failed
        """
        if hash_entry is None:
            return -1, -1
        """
        if self.needs_resize():
            self.double_table_size()
        
        """
        if self.needs_resize():
            self.double_table_size()
        hash_value = self.hash_function(hash_entry.get_position())
        bucket = self.get_ith_bucket(hash_value)
        if bucket is None:
            bucket = b.Bucket(self._bucket_size)
            bucket.add_hash_entry(hash_entry)
            self.set_ith_bucket(bucket, hash_value)
            self._num_non_empty_bucket += 1
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
                self._num_non_empty_bucket += 1
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
                if entry is None:
                    continue
                else:
                    if temp_hash_entry.equals(entry):
                        return entry
            last_entry = bucket.get_ith_entry(self._bucket_size - 1)
            if last_entry is not None:
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
                            # when the bucket stores less than 1 entry, removing 1 entry makes the bucket empty
                            bucket.remove_ith_entry(i)
                            if bucket.get_num_entry_stored() == 0:
                                self.set_ith_bucket(None, hash_value)
                                self._num_non_empty_bucket -= 1
                            return 1
                        # 1b. if last entry and has offset, set last entry to next element and delete next element
                        # if in corresponding bucket and has offset, then must be the last entry
                        else:
                            pointer = element.get_offset()
                            next_element = self._get_hash_entry(pointer[0], pointer[1])
                            self._remove_hash_entry(pointer[0], pointer[1])
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
            if bucket.get_num_entry_stored() == 0:
                self.set_ith_bucket(None, ith_bucket)
                self._num_non_empty_bucket -= 1

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
        self._num_non_empty_bucket = 0
        for bucket in original_hash_table:
            if bucket is None:
                continue
            else:
                for index in range(self._bucket_size):
                    ith_entry = bucket.get_ith_entry(index)
                    if ith_entry is None:
                        continue
                    else:
                        ith_entry.set_offset(None)
                        self.add_hash_entry(ith_entry)
        print("Resize finished.")

    def _in_corresponding_bucket(self, target_entry, b_idx):
        """
        :return: True when hash entry's value is the index of the bucket, else False (it is pointed by an offset)
        """
        return b_idx == self.hash_function(target_entry.get_position())

    def get_num_non_empty_buckets(self):
        return self._num_non_empty_bucket

    def get_volume(self):
        """
        generate tsdf, color volume based on the hash map
        :return: tsdf volume, color volume
        """
        tsdf_volume = np.ones(self._vol_dim).astype(np.float32)
        color_volume = np.zeros(self._vol_dim).astype(np.float32)
        for i in range(self._table_size):
            bucket = self.get_ith_bucket(i)
            for j in range(self._bucket_size):
                entry = bucket.get_ith_entry(j)
                if entry is not None:
                    voxel = entry.get_voxel()
                    if voxel is not None:
                        sdf = voxel.get_sdf()
                        color = voxel.get_color()
                        position = entry.get_position()
                        tsdf_volume[position[0], position[1], position[2]] = sdf
                        color_volume[position[0], position[1], position[2]] = color
        return tsdf_volume, color_volume

    def get_mesh(self):
        """
        source: https://github.com/andyzeng/tsdf-fusion-python
        compute a mesh from the voxel volume using marching cubes
        """
        tsdf_volume, color_volume = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_volume, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_volume[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    def get_point_cloud(self):
        """
        source: https://github.com/andyzeng/tsdf-fusion-python
        extract a point cloud from the voxel volume
        """
        tsdf_volume, color_volume = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_volume, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_volume[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc


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