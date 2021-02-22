import numpy as np

BLUE_CONST = 256 * 256
GREEN_CONST = 256

class Voxel:

    def __init__(self, sdf=1, color=0, weight=0):
        self._sdf = sdf
        self._color = color  # float32 in bgr
        self._weight = weight

    def get_sdf(self):
        return self._sdf

    def get_color(self):
        return self._color

    def integrate(self, new_dist, new_color, obs_weight=1.):
        """
        D’(v) = (D(v)W(v) + w_i(v)d_i(v)) / W(v) + w_i(v)
        W’(v) = W(v) + w_i(v)

        Blue = min(255, (w_old * old_b + obs_weight * new_b) / w_new)
        Green = min(255, (w_old * old_g + obs_weight * new_g) / w_new)
        Red = min(255, (w_old * old_r + obs_weight * new_r) / w_new)

        integrate the input voxel to the currently stored voxel
        :param new_dist: the distance of the voxel to the implicit surface
        :param obs_weight: the weight to assign for the current observation
        """
        w_old = self._weight
        d_old = self._sdf
        w_new = w_old + obs_weight
        d_new = (d_old * w_old + new_dist * obs_weight) / w_new
        self._weight = w_new
        self._sdf = d_new

        old_b = np.floor(self._color / BLUE_CONST)
        old_g = np.floor((self._color - old_b * BLUE_CONST) / GREEN_CONST)
        old_r = self._color - old_b * BLUE_CONST - old_g * GREEN_CONST
        new_b = np.floor(new_color / BLUE_CONST)
        new_g = np.floor((new_color - new_b * BLUE_CONST) / GREEN_CONST)
        new_r = new_color - new_b * BLUE_CONST - new_g * GREEN_CONST
        new_bgr_array = np.minimum(255., np.round(
            [(w_old * old_b + obs_weight * new_b) / w_new,  # blue
             (w_old * old_g + obs_weight * new_g) / w_new,  # green
             (w_old * old_r + obs_weight * new_r) / w_new]))  # red
        self._color = new_bgr_array[0] * BLUE_CONST + new_bgr_array[1] * GREEN_CONST + new_bgr_array[2]


if __name__ == '__main__':
    print("Test hash voxel")
    v = Voxel(1, 2, 3)
    print("Test finished")
