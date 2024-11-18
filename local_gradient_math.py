"""
Here are implemented all the main functions of the local gradient method

Authors: Oleksandr Perederii, Anatolii Kashchuk

Modified by Sanket Patil and Eli Slenders, 2024
"""

import math
from typing import Tuple

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.linalg import lstsq
from sklearn.cluster import DBSCAN


def disk_filter(fltSz: float) -> np.ndarray:
    """
    CREATE DISK FILTER
    :param fltSz: float; radius of disk filter
    :return: h; numpy.array of floats; shape:(2*fltSz+1, 2*fltSz+1)
    """
    crad = math.ceil(fltSz - 0.5)
    [y, x] = np.mgrid[-crad : crad + 1, -crad : crad + 1]
    maxxy = np.maximum(abs(x), abs(y))
    minxy = np.minimum(abs(x), abs(y))

    m1 = (fltSz**2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + (
        fltSz**2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2
    ) * (np.lib.scimath.sqrt(fltSz**2 - (maxxy + 0.5) ** 2)).real
    m2 = (fltSz**2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + (
        fltSz**2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2
    ) * (np.lib.scimath.sqrt(fltSz**2 - (maxxy - 0.5) ** 2)).real

    sgrid = (
        (
            0.5 * (np.arcsin(m2 / fltSz) - np.arcsin(m1 / fltSz))
            + 0.25 * (np.sin(2 * np.arcsin(m2 / fltSz)) - np.sin(2 * np.arcsin(m1 / fltSz)))
        )
        * fltSz**2
        - (maxxy - 0.5) * (m2 - m1)
        + (m1 - minxy + 0.5)
    ) * np.logical_or(
        np.logical_and(
            (fltSz**2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2),
            (fltSz**2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2),
        ),
        np.logical_and(np.logical_and((minxy == 0), (maxxy - 0.5 < fltSz)), (maxxy + 0.5 >= fltSz)),
    )

    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < fltSz**2)
    sgrid[crad, crad] = min(np.pi * fltSz**2, np.pi / 2)

    if (crad > 0) and (fltSz > crad - 0.5) and (fltSz**2 < (crad - 0.5) ** 2 + 0.25):
        print("yes")
        m1 = (np.lib.scimath.sqrt(fltSz**2 - (crad - 0.5) ** 2)).real
        m1n = m1 / fltSz
        sg0 = 2 * (
            (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) * fltSz**2
            - m1 * (crad - 0.5)
        )
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0
        sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0
        sgrid[crad, 1] = sgrid[crad, 1] - sg0
        sgrid[1, crad] = sgrid[1, crad] - sg0

    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    h = sgrid / np.sum(sgrid)
    h = (h - np.min(h)) / (np.max(h - np.min(h)))
    return h


def local_gradient_alloc(
    img_sz: Tuple[int, int], R: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
           PREALLOCATES MATRICES FOR local_gradient
    :param img_sz: tuple; (h, w); h, w - int; height and width of the image for which local gradient is calculated
    :param R: float > 0.5; radius of window filter
    :return: gMatxfft, gMatyfft, sMatfft; three 2d numpy arrays; shape=(h+2*fltSz+1, w+2*fltSz+1);
             2D Fourier transform matrices for calculation of horizontal, vertical and sum of pixels correspondingly
    """
    img_x, img_y = img_sz  # 288, 380
    cR = math.ceil(R - 0.5)
    h = disk_filter(R)
    h = h / np.max(h)

    [g_mat_y, g_mat_x] = np.mgrid[-cR : cR + 1, -cR : cR + 1]
    outsz1, outsz2 = img_x + 2 * cR + 1, img_y + 2 * cR + 1
    gMatxfft = fft2(np.multiply(g_mat_x, h), s=(outsz1, outsz2))
    gMatyfft = fft2(np.multiply(g_mat_y, h), s=(outsz1, outsz2))

    s_mat = np.ones((2 * cR + 1, 2 * cR + 1)) * h
    sMatfft = fft2(s_mat, s=(outsz1, outsz2))
    return gMatxfft, gMatyfft, sMatfft


def local_gradient(
    img: np.ndarray,
    R: float,
    gMatxfft: np.ndarray,
    gMatyfft: np.ndarray,
    sMatfft: np.ndarray,
    thrtype: str,
    thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
           CALCULATES LOCAL GRADIENTS OF THE IMAGE
    :param img: 2d numpy array
    :param R: float > 0.5; radius of window filter
    :param gMatxfft: 2d numpy array
    :param gMatyfft: 2d numpy array
    :param sMatfft: 2d numpy array
           2d Fourier transform matrices for calculation of horizontal, vertical and sum of pixels correspondingly
    :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
    :param thr: non-negative float; threshold value
    :return: g, gradient, 2d numpy array; magnitude of local gradients
             g_x, gradient in x direction, 2d numpy array; horizontal local gradients
             g_y, gradient in y direction, 2d numpy array; vertial local gradients
             g_thr, threshold gradient, 2d numpy array;
             g_mask, binary threshold gradient, 2d numpy array;
             lsq_data, list of three numpy arrays with shapes: (i, 2), (i, 2), (1, i, 1) where i - non-negative int
    """
    img_x, img_y = img.shape
    cR = math.ceil(R - 0.5)
    outsz = np.array([[2 * cR + 1, img_x], [2 * cR + 1, img_y]])
    img = img + 1  # avoid division by zero

    im_fft = fft2(img, s=(outsz[0, 0] + outsz[0, 1], outsz[1, 0] + outsz[1, 1]))
    # sum of all pixels in the area
    im_sum = np.real(ifft2(np.multiply(im_fft, sMatfft)))
    im_sum = im_sum[outsz[0, 0] - 1 : outsz[0, 1], outsz[1, 0] - 1 : outsz[1, 1]]

    # x gradient
    g_x = np.real(ifft2(np.multiply(im_fft, gMatxfft)))
    g_x = np.divide(g_x[outsz[0, 0] - 1 : outsz[0, 1], outsz[1, 0] - 1 : outsz[1, 1]], im_sum)

    # y gradient
    g_y = np.real(ifft2(np.multiply(im_fft, gMatyfft)))
    g_y = np.divide(g_y[outsz[0, 0] - 1 : outsz[0, 1], outsz[1, 0] - 1 : outsz[1, 1]], im_sum)

    # gradient magnitude
    g = np.sqrt(g_x**2 + g_y**2)

    if thrtype == "topfraction":
        cond = np.max(g) / thr
    elif thrtype == "topvalue":
        cond = thr

    # g_size = g.shape
    mask = g > cond
    g_mask = np.where(mask, 1, 0)
    g_thr = np.multiply(g_mask, g)

    c_r = np.argwhere(mask)
    grad = np.vstack((g_x[mask] + c_r[:, 0], g_y[mask] + c_r[:, 1])).T
    v = g[mask]
    lsq_data = [c_r, grad, v.reshape(1, v.shape[0], 1)]
    return g, g_x, g_y, g_thr, g_mask, lsq_data


def lstsqr_lines(
    p1: np.ndarray, p2: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

               LEAST-SQUARE LINE INTERSECTION
    :param p1: 2d numpy array; shape=(i, 2) where i - non-negative int; first points that define lines
    :param p2: 2d numpy array; shape=(i, 2) where i - non-negative int; second points that define lines
    :param w: 3d numpy array; shape=(1, i, 1) where i - non-negative int; line weights
    :return: c_x_y; numpy array; shape=(2, ); Least-squares solution x, y coordinates of the intersection point
             P; numpy array; shape=(i, 2) where i - non-negative int; coordinates of nearest points on each line
             dR; numpy array; shape=(i,) where i - non-negative int; distance from intersection to each line
    """
    n = p2 - p1
    rows, col = n.shape
    n = n / np.sqrt(np.sum(n**2, 1))[:, np.newaxis]
    inn = np.repeat(n.T[:, :, np.newaxis], col, axis=2) * n - np.eye(col)[:, np.newaxis, :]
    inn = inn * w
    r = np.sum(inn, axis=1)
    q = np.matmul(np.vstack((inn[0, :, :], inn[1, :, :])).T, np.hstack((p1[:, 1], p1[:, 0])) + 1)
    c_x_y, _, _, _ = lstsq(r, q, lapack_driver="gelsy")

    # extra outputs
    u = np.sum((np.flip(c_x_y) - (p1 + 1)) * np.fliplr(n), axis=1)
    P = (p1 + 1) + np.repeat(u[:, np.newaxis], col, axis=1) * np.fliplr(
        n
    )  # nearest point on each line
    dR = np.sqrt(
        np.sum((np.flip(c_x_y) - P) ** 2, axis=1)
    )  # distance from intersection to each line
    return c_x_y, P, dR


def get_position_astigmatism(
    img_arr: np.ndarray,
    thrtype: str,
    thr: float,
    R: float,
    G: tuple = None,
    roi: list = None,
    z_pos: bool = True,
    positiveAngle: int = 90,
) -> Tuple[float, float, float]:
    """

            CALCULATE X, Y, Z COORDINATEs OF THE FLUORESCENT PARTICLE IN FLURESCENT MICROSCOPE
    :param img_arr: np.ndarray; 2d image array
    :param thrtype: string; 'topfraction' or 'topvalue'
    :param thr: non-negative float; threshold level
    :param R: float > 0.5; radius of window filter
    :param positiveAngle: int; angle in degrees of positive direction of the particle's image (measured from
                          positive x-axis in counter-clockwise direction)
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :param roi: None by default or list with ints of length 4; region of interest to select an individual fluorophore
                from the image, should be greater than zero and less than corresponding image size
    :param z_pos: bool; whether to calculate z position; True by default
    :return: x, y, z: tuple with 3 floats; x, y, z coordinates of fluorescent particles
    """
    if roi is [None]:
        roi[0], roi[2] = 1, 1
        roi[1], roi[3] = img_arr.shape[0], img_arr.shape[1]

    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        # Precalculate matrices for local gradient calculations
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(
            img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R)
        )

    im_analyze = img_arr[roi[0] - 1 : roi[1], roi[2] - 1 : roi[3]]  # apply region of interest
    # calculate local gradients images
    g, g_x, g_y, g_thr, g_mask, lsq_data = local_gradient(
        im_analyze, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr
    )
    # find center of symmetry of the particle
    c_x_y, P, dR = lstsqr_lines(lsq_data[0], lsq_data[1], lsq_data[2])
    # correct determined positions for the reduction in the image size
    cR = math.ceil(R - 0.5)
    if z_pos:
        # calculate z-value
        z = z_ast(g_thr, g_x, g_y, c_x_y, positiveAngle)
        return c_x_y[0] + cR, c_x_y[1] + cR, z

    return c_x_y[0] + cR, c_x_y[1] + cR, 0.0


def local_gradient_multi(
    img: np.ndarray,
    R: float,
    epsilon: float,
    minpts: int,
    thrtype: str,
    thr: float,
    G: tuple = None,
) -> np.ndarray:
    """
           CALCULATES z-value FOR A FLUORESCENT PARTICLE IN ASTIGMATISM-BASED MICROSCOPY
    :param img: 2d numpy array
    :param R: float > 0.5; radius of window filter
    :param epsilon: float > 0; neighborhood search radius: The maximum distance between two samples for one to be
                    considered as in the neighborhood of the other (DBSCAN)
    :param minpts: int > 0; minimum number of neighbors minpts required to identify a core point
    :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
    :param thr: non-negative float; threshold value
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :return: coord; numpy array; shape=(number_of_clusters, 2); array contains the coordinates of all particles in the
                    image
    """
    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=img.shape, R=abs(R))

    # calculate local gradients images
    g, _, _, _, g_mask, lsq_data = local_gradient(
        img, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr
    )

    # cluster data
    x_y_coord = lsq_data[0]
    idx = DBSCAN(eps=epsilon, min_samples=minpts).fit_predict(
        x_y_coord
    )  # get labels for each point
    n_clusters_ = max(idx) + 1  # len(set(idx)) - (1 if -1 in idx else 0)
    coord = np.zeros((n_clusters_, 2))

    # go through all detected clusters
    for i in range(n_clusters_):
        c_x_y, _, _ = lstsqr_lines(
            lsq_data[0][idx == i, :], lsq_data[1][idx == i, :], lsq_data[2][:, idx == i, :]
        )
        coord[i, :] = c_x_y
    cR = math.ceil(R - 0.5)

    idx = np.argsort(coord[:, 0])  # sort by first coordinate
    coord = coord[idx, :]
    return coord + cR


def detect_trj(c_arr: np.ndarray) -> np.ndarray:
    """
            FUNCTION MODIFIED FOR FINDING LOCATION OF ONE PARTICLE IN EACH IMAGE
            LINKS POSITION DATA INTO TRAJECTORIES

    :param c_arr: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2); N frames, each containing x-y
                  coordinates of n particles
    :return:
            t_xy: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2=output coordinate);
    """

    # t_ext_xy = np.concatenate(c_arr, axis=0)  # (num_of_frames * num_of_particles, 2)
    #
    # t_xy = t_ext_xy[[True]][..., np.newaxis]

    t_xy = np.empty(2, dtype=c_arr.dtype)

    t_xy[0], t_xy[1] = c_arr[0, 0]  # (x, y)
    # t_xy[0] = x
    # t_xy[1] = y

    return t_xy


def _precalc_minV_indV(
    minV: np.ndarray,
    indV: np.ndarray,
    c_arr: np.ndarray,
    k: int,
    dc: int,
    dfr: int,
    nonzero_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """

            PRECALCULATION OF minV AND indV MATRICES

    :param minV: 3d np.ndarray of floats; shape:(number of frames, number of particles, dfr)
    :param indV: 3d np.ndarray of ints; shape:(number of frames, number of particles, dfr)
    :param c_arr: 3d np.ndarray of floats; shape: (num_of_frames, num_of_particles, 2)
    :param k: int > 0; integer in range(1, dfr + 1)
    :param dc: int > 0; maximum distance from the detected particle to look for linked particles in other frames
    :param dfr: int > 0; number of frames to look for linked particles
    :param nonzero_ids: 2d np.ndarray of bool values; shape:(number of frames, number of particles)
                 True value indicates that for a specific (i, j) minV[i, j, :k] array contains
                 all zeros. False - array contains a nonzero value c equal to -1 or > 0
    :return:
            minV, indV: np.ndarray of floats, np.ndarray of ints; shape:(number of frames, number of particles, dfr)
    """
    # if sum of abs values of minV array along last dimension gives nonzero matrix
    if np.all(nonzero_ids == False) or k > dfr:
        return minV, indV
    m1 = c_arr[:, :, np.newaxis, :]
    m2 = c_arr[:, np.newaxis, :, :]

    diff = np.subtract(
        m1[k:, ...],
        m2[:-k, ...],
        where=np.repeat(nonzero_ids[:, :, np.newaxis], 2, axis=2)[:, :, np.newaxis, :],
    )  # get difference only for minV[..., :k] elements that ara all zeros

    diff_sum = np.sqrt(np.sum(diff**2, axis=3))
    m = np.min(diff_sum, axis=2)
    e = m <= dc  # fill minV and indV arrays only if m <= dc
    ij = np.argwhere(np.logical_and(nonzero_ids == True, e))  # and nonzero_ids == True
    minV[ij[:, 0] + k, ij[:, 1], k - 1] = m[ij[:, 0], ij[:, 1]]
    indV[ij[:, 0] + k, ij[:, 1], k - 1] = np.argmin(diff_sum, axis=2)[ij[:, 0], ij[:, 1]]
    # recalculate nonzero_ids for function recursive call with incremented k
    new_nonzero_ids = np.sum(abs(minV[k + 1 :, :, :k]), axis=2) == 0
    return _precalc_minV_indV(minV, indV, c_arr, k + 1, dc, dfr, new_nonzero_ids)


def _first_nonzero(arr: np.ndarray, axis: int, invalid_val: int = -1) -> np.ndarray:
    """

          FIND INDICES OF FIRST NON-ZERO ELEMENT ALONG AXIS

    array_object: np.ndarray
    axis: int
    invalid_val: int, float; the value that will contain the result matrix if all the elements along axis are zeros

    return:
            np.ndarray with indices of first nonzero element of array array_object along axis
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def fast_detect_trj(
    c_arr: np.ndarray, dc: int, dfr: int, Nfr_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

            faster implementation of detect_trj function
            LINKS POSITION DATA INTO TRAJECTORIES

    :param c_arr: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2); N frames, each containing x-y
                  coordinates of n particles
    :param dc: int > 0; maximum distance from the detected particle to look for linked particles in other frames
    :param dfr: int > 0; number of frames to look for linked particles
    :param Nfr_min: int > 0; minimum number of frames in trajectory
    :return:
            t_trj_ids:
            t_frames:
            t_xy: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2=output coordinate);
            t_trj_num:
    """
    num_of_frames = c_arr.shape[0]
    num_of_particles = c_arr.shape[1]
    trj_num = c_arr.shape[1]

    t_ids = np.zeros((num_of_frames, num_of_particles), dtype=int)
    t_frames = np.zeros((num_of_frames, num_of_particles), dtype=int)

    t_ids[0, :] = np.arange(num_of_particles)
    t_frames[0, :] = np.zeros(num_of_particles)

    minV = np.zeros((num_of_frames, num_of_particles, dfr))
    indV = np.zeros((num_of_frames, num_of_particles, dfr), dtype=int)

    r, c, h = np.indices((num_of_frames, num_of_particles, dfr))
    minV[r == h] = -1  # frame + 1 - k <= 0; k in range(1, dfr)

    start_ids = minV[:, :, 0] == 0
    """ c_arr shape: (num_of_frames, num_of_particles, 2) """
    """ minV, indV shape: (num_of_frames, num_of_particles, dfr) """
    minV, indV = _precalc_minV_indV(minV, indV, c_arr, 1, dc, dfr, start_ids[1:, :])

    # get matrice of indexes of the first non-zero elements of minV array along 2 axis and matrice of its values
    frst_nonzero_id = _first_nonzero(minV, axis=2, invalid_val=-1)
    print(minV, indV, frst_nonzero_id.shape)
    values = np.squeeze(np.take_along_axis(minV, frst_nonzero_id[:, :, np.newaxis], 2))

    t_ids = np.zeros((num_of_frames, num_of_particles), dtype=int)
    t_ids[0, :] = np.arange(num_of_particles)
    # set values for cases when frame + 1 - k <= 0 and k == dfr
    print(values.shape)
    t_ids[1:, :][np.where(np.logical_or(values[1:, :] == -1, values[1:, :] == 0))] = np.arange(
        num_of_particles,
        num_of_particles + (values[1:, :] == -1).sum() + (values[1:, :] == 0).sum(),
    )

    t_frames[1:, :] = np.arange(1, num_of_frames)[:, np.newaxis] * np.ones(num_of_particles)

    # set values for the last case
    for fr_part in list(zip(*np.where(t_ids == 0))):
        frame, particle = fr_part
        m = minV[frame, particle, :]
        k = frst_nonzero_id[frame, particle]
        ind = indV[frame, particle, k]
        mtx_temp = t_ids[frame - k - 1, ind]
        t_ids[frame, particle] = mtx_temp

    # trajectory ids for all detected points
    trj_ids = np.sort(np.concatenate(t_ids, axis=0), axis=0)  # (num_of_frames * num_of_particles, )

    # number of frames for each trajectory
    n_frames = np.diff(
        (
            np.diff(np.concatenate((np.array([-1]), trj_ids, np.array([trj_num])), axis=0), axis=0)
            != 0
        ).nonzero()[0]
    )

    # filter trajectories by frame number
    trj_filt = (n_frames >= Nfr_min).nonzero()[0]
    num_of_trj = trj_filt.shape[0]

    t_ext_ids = np.concatenate(t_ids, axis=0)  # (num_of_frames * num_of_particles, )
    t_ext_frames = np.concatenate(t_frames, axis=0)  # (num_of_frames * num_of_particles, )
    t_ext_xy = np.concatenate(c_arr, axis=0)  # (num_of_frames * num_of_particles, 2)

    t_trj_ids, t_frames, t_xy = None, None, None
    t_trj_num = np.empty(num_of_trj)

    for p, t in enumerate(trj_filt):
        mask = t_ext_ids == t
        if p == 0:
            t_trj_ids = t_ext_ids[mask]
            t_frames = t_ext_frames[mask]
            t_xy = t_ext_xy[mask][..., np.newaxis]
        else:
            t_trj_ids = np.vstack((t_trj_ids, t_ext_ids[mask]))
            t_frames = np.vstack((t_frames, t_ext_frames[mask]))
            t_xy = np.concatenate((t_xy, t_ext_xy[mask][..., np.newaxis]), axis=2)
        t_trj_num[p] = t
    return t_trj_ids, t_frames, t_xy, t_trj_num
