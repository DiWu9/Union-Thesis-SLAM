import time

import cv2
import numpy as np

import grid_fusion


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
    print("Estimating voxel volume bounds...")
    cam_intr = np.loadtxt("../datasets/dataset_kitchen/camera-intrinsics.txt", delimiter=' ')
    camera_pose = read_trajectory("../datasets/dataset_stonewall/stonewall_trajectory.log")
    n_imgs = len(camera_pose)
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        depth_im = cv2.imread("../datasets/dataset_stonewall/stonewall_png/depth/%06d.png" % (i+1), -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i]
        view_frust_pts = grid_fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    tsdf_vol = grid_fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("../datasets/dataset_stonewall/stonewall_png/color/%06d.png" % (i+1)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("../datasets/dataset_stonewall/stonewall_png/depth/%06d.png" % (i+1), -1).astype(np.float32)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = camera_pose[i]

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    grid_fusion.meshwrite("../meshes/mesh_grid_demo3.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    grid_fusion.pcwrite("../meshes/pc_grid_demo3.ply", point_cloud)


if __name__ == "__main__":
    main()
