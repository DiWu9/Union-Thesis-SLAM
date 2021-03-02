import cv2
import numpy as np

import grid_fusion
import hash_fusion

import time
import cProfile
import pstats
from pstats import SortKey

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


def read_trajectory(filename):
    """
    Citation: http://redwood-data.org/indoor/fileformat.html
    """
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(mat)
            metastr = f.readline()
    return traj


def main():
    cam_intr = np.loadtxt("../datasets/dataset_kitchen/camera-intrinsics.txt", delimiter=' ')
    camera_pose = read_trajectory("../datasets/dataset_cactusgarden/cactusgarden_trajectory.log")
    n_imgs = 1000
    vol_bnds = np.zeros((3, 2))

    for i in range(1, n_imgs + 1):
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % i, -1).astype(
            np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i - 1]
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    hash_table = hash_fusion.HashTable(vol_bnds, voxel_size=0.02)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(1, n_imgs + 1):
        print("Fusing frame %d/%d" % (i, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/color/%06d.png" % i),
                                   cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % i, -1).astype(
            np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i - 1]

        # Integrate observation into voxel volume (assume color aligned with depth)
        hash_table.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = hash_table.get_mesh()
    grid_fusion.meshwrite("../meshes/mesh_hash_demo2.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    point_cloud = hash_table.get_point_cloud()
    grid_fusion.pcwrite("../meshes/pc_hash_demo2.ply", point_cloud)


def fuse_odd_frames():
    cam_intr = np.loadtxt("../datasets/dataset_kitchen/camera-intrinsics.txt", delimiter=' ')
    camera_pose = read_trajectory("../datasets/dataset_cactusgarden/cactusgarden_trajectory.log")
    n_imgs = len(camera_pose) // 2
    vol_bnds = np.zeros((3, 2))

    for i in range(n_imgs):
        ith_frame = i * 2 + 1
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(
            np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]  # camera pose starts at index 0, which is paired with frame 1
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    hash_table = hash_fusion.HashTable(vol_bnds, voxel_size=0.02)

    total_time = 0
    for i in range(n_imgs):
        tic = time.perf_counter()
        ith_frame = i * 2 + 1
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/color/%06d.png" % ith_frame),
                                   cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(
            np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]

        # Integrate observation into voxel volume (assume color aligned with depth)
        hash_table.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

        toc = time.perf_counter()
        tictoc = round(toc - tic, 2)
        total_time += tictoc
        avg_time = round(total_time / (i + 1), 2)
        print("Integrate frame {} in {} seconds. Avg: {}s/frame".format(ith_frame, tictoc, avg_time))

    verts, faces, norms, colors = hash_table.get_mesh()
    grid_fusion.meshwrite("../meshes/mesh_hash_demo2.ply", verts, faces, norms, colors)

    point_cloud = hash_table.get_point_cloud()
    grid_fusion.pcwrite("../meshes/pc_hash_demo2.ply", point_cloud)


if __name__ == "__main__":
    fuse_odd_frames()
    # main()
