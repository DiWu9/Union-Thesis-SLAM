import time

import cv2
import numpy as np

import grid_fusion
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from memory_profiler import profile


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
    n_imgs = len(camera_pose) // 2
    vol_bnds = np.zeros((3, 2))

    for i in range(n_imgs):
        ith_frame = i * 2 + 1
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame,
                              -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]  # camera pose starts at index 0, which is paired with frame 1
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    tsdf_vol = grid_fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    total_time = 0
    for i in range(n_imgs):
        tic = time.perf_counter()
        ith_frame = i * 2 + 1
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(
            cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/color/%06d.png" % ith_frame),
            cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame,
                              -1).astype(
            np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

        toc = time.perf_counter()
        tictoc = round(toc - tic, 2)
        total_time += tictoc
        avg_time = round(total_time / (i + 1), 2)
        print("Integrate frame {} in {} seconds. Avg: {}s/frame".format(ith_frame, tictoc, avg_time))

    verts, faces, norms, colors = tsdf_vol.get_mesh()
    grid_fusion.meshwrite("../meshes/mesh_grid_demo2.ply", verts, faces, norms, colors)

    point_cloud = tsdf_vol.get_point_cloud()
    grid_fusion.pcwrite("../meshes/pc_grid_demo2.ply", point_cloud)


def hundred_frame_time_profiling():
    """
    grid_fusion: demo2 (cactus garden)
    profiling 100 frames for run-time analysis
    """
    cam_intr = np.loadtxt("../datasets/dataset_kitchen/camera-intrinsics.txt", delimiter=' ')
    camera_pose = read_trajectory("../datasets/dataset_cactusgarden/cactusgarden_trajectory.log")
    n_imgs = len(camera_pose) // 2
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        ith_frame = i * 2 + 1
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]  # camera pose starts at index 0, which is paired with frame 1
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    tsdf_vol = grid_fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
    for i in range(100):
        ith_frame = i * 2 + 1
        color_image = cv2.cvtColor(cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/color/%06d.png" % ith_frame), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)


@profile
def hundred_frame_space_profiling():
    """
    grid_fusion: demo2 (cactus garden)
    profiling 100 frames for memory analysis
    """
    cam_intr = np.loadtxt("datasets/dataset_kitchen/camera-intrinsics.txt", delimiter=' ')
    camera_pose = read_trajectory("datasets/dataset_cactusgarden/cactusgarden_trajectory.log")
    n_imgs = len(camera_pose) // 2
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        ith_frame = i * 2 + 1
        depth_im = cv2.imread("datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    tsdf_vol = grid_fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
    for i in range(100):
        ith_frame = i * 2 + 1
        color_image = cv2.cvtColor(cv2.imread("datasets/dataset_cactusgarden/cactusgarden_png/color/%06d.png" % ith_frame), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("datasets/dataset_cactusgarden/cactusgarden_png/depth/%06d.png" % ith_frame, -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i * 2]
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)


def profile_main():
    graphviz = GraphvizOutput()
    graphviz.output_file = '../call_graph/demo2_100_frame_call_graph/call_graph_grid_demo2.png'
    with PyCallGraph(output=graphviz):
        hundred_frame_time_profiling()


if __name__ == "__main__":
    hundred_frame_space_profiling()
    # profile_main()
    # main()
