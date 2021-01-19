import numpy as np


class Voxel:

    def __init__(self, sdf, color, weight):
        self.sdf = sdf
        self.color = color
        self.weight = weight

    def integrate(self, new_dist, obs_weight=1.):
        """
        D’(v) = (D(v)W(v) + w_i(v)d_i(v)) / W(v) + w_i(v)
        W’(v) = W(v) + w_i(v)

        integrate the input voxel to the currently stored voxel
        :param new_dist: the distance of the voxel to the implicit surface
        :param obs_weight: the weight to assign for the current observation
        :return:
        """
        w_old = self.weight
        d_old = self.sdf
        w_new = w_old + obs_weight
        d_new = (d_old * w_old + new_dist * obs_weight) / w_new
        self.weight = w_new
        self.sdf = d_new


if __name__ == '__main__':
    print("Test hash voxel")
    v = Voxel(1, 2, 3)
    print("Test finished")
