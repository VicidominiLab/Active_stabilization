"""Base class ParticleDetector Provides methods for calculation particle coordinates in astigmatism and double-helix
fluorescent microscopy and in brightfield microscopes.

Authors: Oleksandr Perederii, Anatolii Kashchuk

Modified by Sanket Patil and Eli Slenders, 2024
"""

from typing import Callable

import numpy as np
from numpy import ndarray

from validators import validate
from local_gradient_math import (
    local_gradient_alloc,
    get_position_astigmatism,
    local_gradient_multi,
    detect_trj,
)


class ParticleDetector:
    @staticmethod
    def _normalize_roi_for_image(roi, img):
        """
                PREPROCESS roi ARGUMENT
        :param roi: list with ints of length 4; region of interest to select an individual fluorophore from the image,
                    should be greater than zero and less than corresponding image size
        :param img: an image object <<class PIL.Image.Image>>
        :return: roi: list with ints of length 4 that corresponds to the selected area of the image or coordinates
                      of the entire image otherwise
        """
        roi = list(map(int, roi))
        fst_img = img  # PIL.ImageSequence.Iterator(img)[0]
        fst_dim, scd_dim = np.array(fst_img).shape[0], np.array(fst_img).shape[1]
        # if roi size are incorrect - set them to full image size
        if (
            (roi == [0, 0, 0, 0])
            or (0 < roi[1] < roi[0] < fst_dim)
            or (0 < roi[3] < roi[2] < scd_dim)
        ):
            roi[0], roi[2] = 1, 1
            roi[1], roi[3] = fst_dim, scd_dim
        return roi

    @staticmethod
    def _process_image(img, callback, *args, **kwargs):
        """
        :param img: an image object <<class PIL.Image.Image>>
        :param callback: method
        :param args:
        :param kwargs:
        :return: x, y, zV, t; np.ndarray-s with x, y, z coordinates of the particles and execution time
        """
        x, y, zV = np.array([]), np.array([]), np.array([])
        # for page in PIL.ImageSequence.Iterator(img):
        im = img  # np.array(page)
        if im.ndim == 3:
            im = im[:, :, 0]
        x_i, y_i, z_i = callback(im, *args, **kwargs)
        x = np.append(x, x_i)
        y = np.append(y, y_i)
        zV = np.append(zV, z_i)
        return (x, y, zV)

    @classmethod
    def _process(
        cls,
        input_image: np.ndarray,
        thrtype: str,
        thr: float,
        R: float,
        roi: list,
        z_pos: bool,
        callback: Callable,
        **kwargs,
    ):
        """
        :param filename: string; path to the image file
        :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
        :param thr: non-negative float; threshold value
        :param R: float > 0.5; radius of window filter
        :param roi: list with ints of length 4; region of interest to select
        :param dz: float; z step between images
        :param z0: int; first image position
        :param z_pos: bool; whether to calculate z position
        :param draw: bool; whether to show a drawing
        :param callback: function
        :param kwargs:
        :return: x, y, zV, t; np.ndarray-s with x, y, z coordinates of the particles and execution time
        """
        img = input_image  # PIL.Image.open(filename)
        roi = cls._normalize_roi_for_image(roi, img)

        # Precalculate matrices for local gradient calculations
        matrices = local_gradient_alloc(img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R))

        x, y, zV = cls._process_image(
            img,
            callback,
            thrtype,
            thr,
            R,
            matrices,
            roi,
            z_pos,
            # img=img,
            # callback=callback,
            **kwargs,
        )
        return x, y, zV

    @classmethod
    @validate
    def get_pos_astigmatism(
        cls,
        input_image: np.ndarray,
        thr: float,
        R: float,
        positiveAngle: int,
        roi: list,
        thrtype: str = "topfraction",
        z_pos: bool = True,
    ) -> [float, float, float]:
        """
                CALCULATE X, Y, Z COORDINATES OF FLUORESCENT PARTICLE
        :param input_image: ndarry of input image
        :param thrtype: string; 'topfraction' or 'topvalue'
        :param thr: non-negative float; threshold level
        :param R: float > 0.5; radius of window filter
        :param roi: list with ints of length 4; region of interest to select an individual fluorophore from the image,
                                                should be greater than zero and less than corresponding image size
        :param positiveAngle: int; angle in degrees of positive direction of the particle's image (measured from
                                positive x-axis in counter-clockwise direction)
        :param z_pos: bool; whether to calculate z position
        :return: x, y, zV: 1d numpy arrays; x, y, z coordinates of fluorescent particles
        """
        x, y, zV = cls._process(
            input_image,
            thrtype,
            thr,
            R,
            roi,
            z_pos,
            callback=get_position_astigmatism,
            positiveAngle=positiveAngle,
        )
        return float(x), float(y), float(zV)

    @classmethod
    @validate
    def get_multi_trajectory(
        cls, cropped_roi, thr: float, R: float, epsilon: float, minpts: int
    ) -> ndarray:
        """
                FUNCTION MODIFIED FOR FINDING LOCATION OF ONE PARTICLE IN EACH IMAGE
                LINKS POSITION DATA INTO TRAJECTORIES
        @param cropped_roi: cropped roi from the input image
        @param thr: non-negative float; threshold level
        @param R: float > 0.5; radius of window filter
        @param epsilon: float > 0; neighborhood search radius: The maximum distance between two samples for one to be
                        considered as in the neighborhood of the other (DBSCAN)

        @return: np.ndarray-S; trj_id, frames, output, trj_num; shapes: (num_of_trajectories, num_of frames),
                 (num_of_trajectories, num_of frames), (num_of_trajectories, num_of frames, 2), (num_of_trajectories, )
        """

        coord_z = np.array(
            [
                local_gradient_multi(
                    cropped_roi, R=R, epsilon=epsilon, minpts=minpts, thr=thr, thrtype="topfraction"
                )
            ]
        )

        return detect_trj(coord_z)
