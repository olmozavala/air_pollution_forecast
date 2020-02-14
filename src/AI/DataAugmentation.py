import numpy as np
import preproc.UtilsPreproc as utils
from scipy.ndimage.filters import gaussian_filter


def flipping(imgs, ctrs, flip_axis):
    """ Flips images and contours in the specified axis:"""
    tot_imgs = len(imgs)
    tot_ctrs = len(ctrs)

    # Flips the images
    for c_img_idx in range(tot_imgs):
        imgs[c_img_idx] = np.flip(imgs[c_img_idx], axis=flip_axis)
    # Flips the ctrs
    for c_ctr_idx in range(tot_ctrs):
        ctrs[c_ctr_idx] = np.flip(ctrs[c_ctr_idx], axis=flip_axis)

    return imgs, ctrs


def gaussblur_3d(imgs, sigma_size = 3):
    sigma = sigma_size * np.random.random()  # Blur from 0 to 2

    tot_imgs = len(imgs)
    # print("Gaussian blur, sigma: {}".format(sigma))
    for i in range(tot_imgs):
        imgs[i] = gaussian_filter(imgs[i], sigma=sigma, order=0, mode='mirror')
    return imgs


def shifting_3d(imgs, ctrs, max_shift_perc=.1):
    shiftSize = int(imgs.shape[3] * max_shift_perc * (np.random.random() * 2 - 1))
    shiftAxis = np.random.randint(0, 3)
    # print("Shifting {} Axis {}".format(shiftSize, shiftAxis))
    if shiftSize != 0:
        for i in range(len(imgs)):
            imgs[i, 0, :, :, :] = utils.shift3D(imgs[i, 0, :, :, :], shiftSize, shiftAxis)
            imgs[i, 1, :, :, :] = utils.shift3D(imgs[i, 1, :, :, :], shiftSize, shiftAxis)
            imgs[i, 2, :, :, :] = utils.shift3D(imgs[i, 2, :, :, :], shiftSize, shiftAxis)

        for i in range(len(ctrs)):
            ctrs[i, 0, :, :, :] = utils.shift3D(ctrs[i, 0, :, :, :], shiftSize, shiftAxis)

    return imgs, ctrs
