import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from pylablib.devices import Thorlabs
from matplotlib.patches import Rectangle


numFrames_for_z: int = 5

pixel_to_nm_conv: float = 34

tracking_region: float = 50  # Pixels
movement_range: float = 1
inertia_correction_factor: float = 2000  # range in the 1000s because the number needs to converted in nm from

image_save_interval: int = 1000  # Number of frames after which the image is saved (Both XY and Z)

# Setting the limits for the plots in addition to fixing number of points to plot
nPoints = 300

# Points to hold before saving the data to a csv file
dump_points = 1000


# Boolean properties
plotting: bool = True
data_saving: bool = True
save_images: bool = False
PID_stabilization: bool = False
closed_loop_feedback: bool = True
non_PID_stabilization: bool = not PID_stabilization

# Camera Serial Numbers
xy_camera_serial: str = "22424"
z_camera_serial: str = "09490"


def get_exposure_time() -> tuple:
    with (Thorlabs.ThorlabsTLCamera(serial=xy_camera_serial) as xy_cam, Thorlabs.ThorlabsTLCamera(serial=z_camera_serial) as z_cam):
        exposure_time_xy = xy_cam.get_exposure()
        exposure_time_z = z_cam.get_exposure()

    return exposure_time_xy, exposure_time_z


# New paths for saving images and csv files
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

if not os.path.exists(f"{os.getcwd()}/Localization_data"):
    os.makedirs(f"{os.getcwd()}/Localization_data")


class ROISelector:
    def __init__(self, roi_size, save_path, title):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Double-click to select ROIs")
        self.roi_size = roi_size
        self.rois = []
        self.drawing = False
        self.current_rect = None
        self.title = title
        self.save_path = save_path
        self.roi_counter = 1  # Counter for numbering ROIs

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_press(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            width, height = self.roi_size, self.roi_size
            self.current_rect = Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor="r", facecolor="none")
            self.ax.add_patch(self.current_rect)
            self.drawing = True

    def on_release(self, event):
        if self.drawing:
            roi = {"coordinates": self.get_rect_coordinates(), "number": self.roi_counter}
            self.rois.append(roi)
            self.roi_counter += 1
            self.current_rect = None
            self.drawing = False
            plt.draw()

    def on_motion(self, event):
        if self.drawing:
            self.current_rect.set_width(self.roi_size)
            self.current_rect.set_height(self.roi_size)
            self.current_rect.set_x(event.xdata - self.roi_size / 2)
            self.current_rect.set_y(event.ydata - self.roi_size / 2)
            plt.draw()

    def get_rect_coordinates(self):
        return self.current_rect.get_x() + self.roi_size / 2, self.current_rect.get_y() + self.roi_size / 2, self.roi_size, self.roi_size

    def on_close(self, event):
        self.save_image_with_boxes()

    def save_image_with_boxes(self):
        self.ax.set_title(f"{self.title}")
        plt.draw()

        for roi_data in self.rois:
            rect = Rectangle(
                (roi_data["coordinates"][0] - roi_data["coordinates"][2] / 2, roi_data["coordinates"][1] - roi_data["coordinates"][3] / 2),
                roi_data["coordinates"][2],
                roi_data["coordinates"][3],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            # Add number to the center of the rectangle
            self.ax.text(
                (roi_data["coordinates"][0] - self.roi_size / 2) - 5,
                (roi_data["coordinates"][1] - self.roi_size / 2) - 5,
                str(roi_data["number"]),
                color="r",
                fontsize=10,
                ha="center",
                va="center",
            )

        plt.savefig(self.save_path, dpi=500, bbox_inches="tight")
        print(f"Image with numbered boxes saved to {self.save_path}")

    def select_rois(self, image):
        self.image = image
        # turn off the axis ticks and labels
        self.ax.imshow(image, cmap="gray")
        self.ax.set_axis_off()
        plt.show()
        return self.rois


def get_path_for_csv(folder_name=dt_string) -> str:
    # Create a new folder for saving the data
    if not os.path.exists(f"{os.getcwd()}/Localization_data/{dt_string}"):
        os.makedirs(f"{os.getcwd()}/Localization_data/{dt_string}")

    # Create a new folder for saving the csv files
    newpath_for_csv = f"{os.getcwd()}/Localization_data/{folder_name}/CSV"
    if not os.path.exists(newpath_for_csv):
        os.makedirs(newpath_for_csv)
    return newpath_for_csv


def get_path_for_images(folder_name=dt_string) -> str:
    # Create a new folder for saving the images
    newpath_for_images = f"{os.getcwd()}/Localization_data/{folder_name}/Images"
    if not os.path.exists(newpath_for_images):
        os.makedirs(newpath_for_images)
    return newpath_for_images


def delete_temporary_image() -> None:
    # Delete the temporary image created for ROI selection
    try:
        os.remove("color_img.jpg")
    except OSError:
        pass


def create_csv(filename) -> None:
    with open(filename, "w") as file:
        pass


def save_config(config: dict) -> None:
    """
    Save the configuration file for the system

    @param filename: Name of the configuration file
    @param config: Configuration dictionary
    """
    with open(f"{os.getcwd()}/config_dict.pkl", "wb") as file:
        pickle.dump(config, file)

    path = f"{os.getcwd()}/Localization_data/{dt_string}"
    with open(f"{path}/config_dict.pkl", "wb") as file:
        pickle.dump(config, file)


def load_config(filename: str) -> dict:
    """
    Load the configuration file for the system

    @param filename: Name of the configuration file

    @return: Configuration dictionary
    """
    with open(filename, "rb") as file:
        config = pickle.load(file)

    return config


def save_to_csv(filename, data, mode) -> None:
    """
    Save the input data to a csv file

    @param filename: Name and path of the file to be saved
    @param data: Data to be saved
    @param mode: Mode of saving the file ("x" to create new file, "a" to append to existing file)
    """
    pd.DataFrame(data).to_csv(filename, mode=mode, encoding="utf-8", index=False, header=False)


def running_mean(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculates the running mean of the input data with the input window size

    @param data: time series data
    @param window_size: number of frames to be averaged over

    @return: Running mean of input data
    """
    assert window_size > 0
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def weights_for_average(scaling_factor: float = 1.3, num_points: int = 10, flip: bool = True) -> np.ndarray:
    """
    Calculates the weights for the weighted average based on the scaling factor and number of points

    @param scaling_factor: Scaling factor for the weights
    @param num_points: Number of points to be averaged
    @param flip: Boolean to flip the weights

    @return: Weights for the weighted average
    """
    k = np.array([(1 / scaling_factor) ** i for i in range(num_points)])
    k = k / np.sum(k)
    if flip:
        return np.flip(k)
    else:
        return k


@njit
def array_shift(array_object: np.ndarray, shift_elements: int = -1, fill_value: float = np.nan) -> np.ndarray:
    """
    @param array_object: Array to be shifted (1D)
    @param shift_elements: Number of elements to be shifted (positive for right shift, negative for left shift)
    @param fill_value: Fill value for the shifted elements

    @return: Resulting shifted array
    """

    result = np.empty_like(array_object)
    if shift_elements > 0:
        result[:shift_elements] = fill_value
        result[shift_elements:] = array_object[:-shift_elements]
    elif shift_elements < 0:
        result[shift_elements:] = fill_value
        result[:shift_elements] = array_object[-shift_elements:]
    else:
        result[:] = array_object
    return result


@njit
def array_shift_2d(array_object: np.ndarray, shift_elements: int = -1, fill_value: np.ndarray = None) -> np.ndarray:
    """
    @param array_object: Array to be shifted (2D)
    @param shift_elements: Number of elements to be shifted (positive for right shift, negative for left shift)
    @param fill_value: Fill value for the shifted elements

    @return: Resulting shifted array (2D)
    """

    result = np.empty_like(array_object)
    if shift_elements > 0:
        result[:shift_elements, :] = fill_value
        result[shift_elements:, :] = array_object[:-shift_elements, :]
    elif shift_elements < 0:
        result[shift_elements:, :] = fill_value
        result[:shift_elements, :] = array_object[-shift_elements:, :]
    else:
        result[:, :] = array_object
    return result


@njit(parallel=True, fastmath=True)
def array_shift_3d(array_object: np.ndarray, shift_elements: int = -1, fill_value: np.ndarray = None) -> np.ndarray:

    # result = array_object[..., :shift_elements]
    # result = np.concatenate((result, fill_value[...,np.newaxis]), axis=-1)
    #
    # return result

    result = np.empty_like(array_object)  # Slowest step in the code
    if shift_elements > 0:
        result[:, :, :shift_elements] = fill_value[..., np.newaxis]
        result[:, :, shift_elements:] = array_object[:, :, :-shift_elements]
    elif shift_elements < 0:
        result[:, :, shift_elements:] = fill_value[..., np.newaxis]
        result[:, :, :shift_elements] = array_object[:, :, -shift_elements:]
    else:
        result[:, :, :] = array_object
    return result


# XY Camera ROI selection
def roi_select_for_acquisition(serial=xy_camera_serial) -> list:
    with Thorlabs.ThorlabsTLCamera(serial=serial) as cam:  # to close the camera automatically
        cam.set_roi(0, 1440, 0, 1080, 1, 1)
        print("XY Camera ROI = ", cam.get_roi())

        cam.start_acquisition()  # start acquisition (automatically sets it up as well)
        cam.wait_for_frame()  # wait for the next available frame

        region_select_image = cam.read_newest_image()
        region_select_image = (region_select_image / np.max(region_select_image)) * 255

        cv2.imwrite("color_img.jpg", region_select_image)
        im_color = cv2.imread("color_img.jpg", cv2.IMREAD_COLOR)
        ROI = cv2.selectROIs("Select Rois for XY Plane", im_color)
        cv2.destroyAllWindows()

        delete_temporary_image()

        ROI = ROI[0]
        print(ROI)
        cam.set_roi(ROI[0], ROI[0] + ROI[2], ROI[1], ROI[1] + ROI[3])

        cam_roi = cam.get_roi()
        print("XY Camera ROI changed to ", cam_roi)

    return [[cam_roi[1] - cam_roi[0], cam_roi[3] - cam_roi[2]], cam_roi]


# Z Camera ROI selection
def Z_image_roi_for_FFT(serial=z_camera_serial) -> list:
    with Thorlabs.ThorlabsTLCamera(serial=serial) as cam:
        cam.set_roi(0, 1440, 0, 1080, 1, 1)
        print("Z Camera ROI = ", cam.get_roi())
        # cam.set_exposure(exposure_time_z)

        cam.start_acquisition()  # start acquisition (automatically sets it up as well)
        cam.wait_for_frame()  # wait for the next available frame

        region_select_image = cam.read_newest_image()
        region_select_image = (region_select_image / np.max(region_select_image)) * 255

        cv2.imwrite("color_img.jpg", region_select_image)
        im_color = cv2.imread("color_img.jpg", cv2.IMREAD_COLOR)
        ROI = cv2.selectROIs("Select Roi for Z FFT autocorrelation Measurements", im_color)
        cv2.destroyAllWindows()

        delete_temporary_image()

        ROI = ROI[0]

        # check if size of roi is odd or even
        ROI[2] = ROI[2] + 1 if ROI[2] % 2 != 0 else ROI[2]
        ROI[3] = ROI[3] + 1 if ROI[3] % 2 != 0 else ROI[3]

        # set minimum size of roi to 150 pixels
        ROI[2] = 150 if ROI[2] < 150 else ROI[2]
        ROI[3] = 150 if ROI[3] < 150 else ROI[3]

        cam.set_roi(ROI[0], ROI[0] + ROI[2], ROI[1], ROI[1] + ROI[3])
        cam_roi = cam.get_roi()
        print(f"Z autocorrelation ROI = {cam_roi}")

        return [[cam_roi[1] - cam_roi[0], cam_roi[3] - cam_roi[2]], cam_roi]


# XY Camera related functions
def get_rois_from_cam_image(camera_exposure: float, serial=xy_camera_serial) -> list:
    """
    @param camera_exposure: Exposure time in seconds to feed the camera
    @param serial: Serial number of the camera to be connected

    @return: List of user selected regions of interest
    """
    if camera_exposure <= 0:
        raise ValueError("Exposure time should be greater than 0")
    if camera_exposure > 0.03:
        raise FutureWarning("Exposure time may be too high - Can cause saturation in the image and slow down the process")

    with Thorlabs.ThorlabsTLCamera(serial=serial) as cam:  # to close the camera automatically
        cam.set_exposure(camera_exposure)
        cam.start_acquisition()  # start acquisition (automatically sets it up as well)
        cam.wait_for_frame()  # wait for the next available frame

        roi_select_image = cam.read_newest_image()  # h5data[:, :, 0]

        roi_selector = ROISelector(tracking_region, f"{get_path_for_images()}/XY_rois.png", "XY_ROIs")
        selected_rois = roi_selector.select_rois(roi_select_image)

        ROIs = []
        for roi_data in selected_rois:
            ROIs.append(list(roi_data["coordinates"]))

    return ROIs


def calculate_astigmatism(image_for_fft: np.ndarray, fft_pixels_in_x: int = 130, fft_pixels_in_y: int = 9) -> float:
    """
    @param image_for_fft: Image to be used for FFT calculation
    @param fft_pixels_in_x: Number of pixels to be used in the FFT calculation in the X direction
    @param fft_pixels_in_y: Number of pixels to be used in the FFT calculation in the Y direction

    @return: Astigmatism value
    """
    # check if fft_pixels_in_x and fft_pixels_in_y are integers and greater than 0
    if not isinstance(fft_pixels_in_x, int) or not isinstance(fft_pixels_in_y, int):
        raise ValueError("fft_pixels_in_x and fft_pixels_in_y should be integers")
    if fft_pixels_in_x <= 0 or fft_pixels_in_y <= 0:
        raise ValueError("fft_pixels_in_x and fft_pixels_in_y should be greater than 0")

    fft1 = np.fft.fft2(image_for_fft)

    # Cross correlating the same image with itself
    fft2 = fft1.copy()
    fft2 = np.conj(fft2)

    # Element wise multiplication
    result = fft1 * fft2
    result_img = np.fft.ifft2(result)
    result_img = np.abs(result_img)

    image_shifted = np.fft.fftshift(result_img)

    fft_shape = np.shape(image_shifted)
    center_for_Z_FFT = [(fft_shape[0] / 2), (fft_shape[1] / 2)]

    horizontal_profile = np.sum(
        image_shifted[
            int(center_for_Z_FFT[0] - int(fft_pixels_in_y / 2)) : int(center_for_Z_FFT[0] + int(fft_pixels_in_y / 2)),
            int(center_for_Z_FFT[1] - int(fft_pixels_in_x / 2)) : int(center_for_Z_FFT[1] + int(fft_pixels_in_x / 2)),
        ]
    )
    vertical_profile = np.sum(
        image_shifted[
            int(center_for_Z_FFT[0] - int(fft_pixels_in_x / 2)) : int(center_for_Z_FFT[0] + int(fft_pixels_in_x / 2)),
            int(center_for_Z_FFT[1] - int(fft_pixels_in_y / 2)) : int(center_for_Z_FFT[1] + int(fft_pixels_in_y / 2)),
        ]
    )

    astigmatism = vertical_profile / horizontal_profile

    return astigmatism


def get_fft_calibration_value(z_position: float,exposure_time_z, calibration_z_nm: float = 50, num_z_points: int = 5) -> float:
    from pi_stage_controller import send_axis_position

    """
    Function for calculating the FFT calibration value
    @param z_position: Current Z position of the stage
    @param calibration_z_nm: Step size for the calibration in nm -> 50 means 50*5 = 250 nm
    @param num_z_points: Number of points to be taken for the calibration. Should be odd number for the center to be the original position.

    @return: Astigmatism value list
    """

    if num_z_points % 2 == 0:
        raise FutureWarning("Number of Z points is even - Reference point will not be included in the calibration.")

    astig_list = []
    with Thorlabs.ThorlabsTLCamera(serial=z_camera_serial) as cam:
        cam.set_exposure(exposure_time_z)
        cam.start_acquisition()

        for z_pos in (pbar := tqdm(range(num_z_points))):
            pbar.set_description("Calibrating Z axis FFT at {} nm".format((z_pos - int(num_z_points / 2)) * calibration_z_nm))

            send_axis_position(2, float(z_position + ((z_pos - int(num_z_points / 2)) * calibration_z_nm) / 1000))

            # Sleep time for the stage to move in position (stage can sometimes not be in place when the image is taken)
            sleep(0.5)

            cam.wait_for_frame()
            image_z_for_fft = cam.read_newest_image()

            astig = calculate_astigmatism(image_z_for_fft)
            astig_list.append(astig)

    # Calculating FFT calibration value - AU per nm
    fft_calib_value_per_nm = float(np.polyfit(np.arange(len(astig_list)), astig_list, 1)[0] / calibration_z_nm)

    # Saving the FFT calibration value to a text file for future use
    np.savetxt(f"{get_path_for_csv()}/astigmatism_values.txt", astig_list)
    np.savetxt(f"{get_path_for_csv()}/Z_axis_FFT_calibration.txt", [float(fft_calib_value_per_nm)])

    return fft_calib_value_per_nm


def controller_update_with_PID(
    axis: str,
    pid_object: object,
    current_frame: int,
    axis_average: np.ndarray,
    current_axis_position: float,
    numframes_for_z: int,
    fft_calibration_value_z: float = 1,
) -> float:
    from pi_stage_controller import send_axis_position

    # TODO: Make the Z calib properly with more variables
    """
    Axis PID stabilization code --
    Updates the controller position based on the PID control Algorithm

    @param axis_average: Array containing the average movement of all particles in the frame
            (autocorr_values_Z for Z axis in main code)
    @param axis: axis to be stablilized (X, Y, or Z)
    @param current_axis_position: Current axis position of the stage
    @param current_frame: Current frame number (z_frame_count for the Z axis in main code)
    @param numFrames_for_z:  of frames to for Z axis stabilization
    @param pid_object: PID object for the axis to be stabilized
    @param fft_calibration_value_z: FFT Calibration value for Z axis

    @return: Updated axis position value using PID
    """
    pid_object.set_current_frame(current_frame)

    if str(axis).upper() == "X":
        expected_axis_position_setpoint = current_axis_position - (
            (axis_average[current_frame - int(current_frame / numframes_for_z)] - axis_average[0]) / 3000
        )
    elif str(axis).upper() == "Y":
        expected_axis_position_setpoint = current_axis_position + (
            (axis_average[current_frame - int(current_frame / numframes_for_z)] - axis_average[0]) / 3000
        )
    elif str(axis).upper() == "Z":
        expected_axis_position_setpoint = current_axis_position - ((axis_average[current_frame] / fft_calibration_value_z) / 2000)

    pid_object.set_SetPoint(expected_axis_position_setpoint)
    pid_object.update(current_axis_position=current_axis_position, current_frame=current_frame)
    position_difference_to_be_changed_on_axis = pid_object.output

    print(expected_axis_position_setpoint, current_axis_position, position_difference_to_be_changed_on_axis)

    if str(axis).upper() == "X":
        send_axis_position(0, current_axis_position + position_difference_to_be_changed_on_axis)
    elif str(axis).upper() == "Y":
        send_axis_position(1, current_axis_position + position_difference_to_be_changed_on_axis)
    elif str(axis).upper() == "Z":
        send_axis_position(2, current_axis_position + position_difference_to_be_changed_on_axis)

    # if str(axis).upper() == 'X' or 'Z':
    #     current_axis_position += position_difference_to_be_changed_on_axis
    # elif str(axis).upper() == 'Y':
    current_axis_position += position_difference_to_be_changed_on_axis

    return current_axis_position
