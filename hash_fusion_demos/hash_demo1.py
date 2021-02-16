import time

import cv2
import numpy as np

import grid_fusion
import hash_fusion

import time
import cProfile
import pstats
from pstats import SortKey


def one_frame_profiling():
    n_imgs = 1000
    cam_intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        depth_im = cv2.imread("../data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % (i))
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    hash_table = hash_fusion.HashTable(vol_bnds, voxel_size=0.02)

    # fuse frame 0 for testing
    color_image = cv2.cvtColor(cv2.imread("../data/frame-%06d.color.jpg" % 0), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("../data/frame-%06d.depth.png" % 0, -1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % 0)
    hash_table.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)


def ten_frame_profiling():
    n_imgs = 1000
    cam_intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        depth_im = cv2.imread("../data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % (i))
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    hash_table = hash_fusion.HashTable(vol_bnds, voxel_size=0.02)

    total_time = 0
    # Loop through the first 10 RGB-D images and fuse them together
    for i in range(10):
        tic = time.perf_counter()

        color_image = cv2.cvtColor(cv2.imread("../data/frame-%06d.color.jpg" % (i)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % (i))
        hash_table.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

        toc = time.perf_counter()
        tictoc = round(toc - tic, 2)
        total_time += tictoc
        avg_time = round(total_time / (i + 1), 2)
        print("Integrate frame {} in {} seconds. Avg: {}s/frame".format(i + 1, tictoc, avg_time))
        """
        print("Frame {}: num occupied buckets: {}; load factor: {}; collisions: {}; collision rate: {}".format(
            i + 1,
            hash_table.get_num_non_empty_bucket(),
            hash_table.get_load_factor(),
            hash_table.get_num_collisions(),
            hash_table.get_num_collisions() / hash_table.get_num_non_empty_bucket()
        ))
        """


def main():
    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    cam_intr = np.loadtxt("../data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = cv2.imread("../data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % (i))  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    print("Initializing hash table...")
    hash_table = hash_fusion.HashTable(vol_bnds, voxel_size=0.02, map_size=2000000)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("../data/frame-%06d.color.jpg" % (i)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("../data/frame-%06d.pose.txt" % (i))

        # Integrate observation into voxel volume (assume color aligned with depth)
        hash_table.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = hash_table.get_mesh()
    grid_fusion.meshwrite("mesh_hash_demo1.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = hash_table.get_point_cloud()
    grid_fusion.pcwrite("pc_hash_demo1.ply", point_cloud)


def test_hash_function():
    positions = [[333, 234, 241], [342, 234, 241], [332, 234, 242]]
    hash_map = hash_fusion.HashTable([[-4.22106438, 3.86798203], [-2.6663104, 2.60146141], [0., 5.76272371]], 0.02,
                                     100000, False)
    for position in positions:
        hash_value = hash_map.hash_function(position)
        print(hash_value)


def profile_function_write_file(function_name, filename):
    cProfile.run(function_name, filename)


def read_profile_file(filename, top_n_functions):
    p = pstats.Stats(filename)
    p.sort_stats(SortKey.TIME).print_stats(top_n_functions)


if __name__ == "__main__":
    # profile_function_write_file('test_hash_function()', 'cProfile/stats_hash_function')
    # read_profile_file('cProfile/stats_hash_function', 10)
    # profile_function_write_file('one_frame_profiling()', 'cProfile/stats_one_frame')
    # profile_function_write_file('ten_frame_profiling()', 'cProfile/stats_ten_frame')
    # one_frame_profiling()
    # read_profile_file('cProfile/stats_one_frame', 20)
    # read_profile_file('cProfile/stats_ten_frame', 20)
    main()
    # ten_frame_profiling()
