import multiprocessing as mp

# Local Imports
from particle_detector import ParticleDetector
from mp_shared_array import MemorySharedNumpyArray


class Text_Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# Initializing the position calculation function
def position_calc(img):
    return ParticleDetector().get_multi_trajectory(img, thr=2, R=15, epsilon=3, minpts=7)


def camera_capture(
    serial: str,
    exposure_time: float,
    shared_image_data: MemorySharedNumpyArray,
    frame_counter: mp.Value,
    process_start_switch: mp.Event,
    frame_ready_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """
    Capture the images from the Thorlabs Camera
    @param serial: Serial number of the camera (Thorlabs) (str)
    @param exposure_time: Exposure time for the camera (float)
    @param shared_image_data: Shared memory array for the image data (MemorySharedNumpyArray)
    @param frame_counter: Frame counter for the camera (mp.Value)
    @param process_start_switch: Process start switch for the camera (mp.Event)
    @param frame_ready_event: Frame ready event for the camera (mp.Event)
    @param stop_event: Stop event for the camera (mp.Event)

    @return: None
    """

    from pylablib.devices import Thorlabs

    # Connecting to the Thorlabs Camera and starting acquisition
    cam = Thorlabs.ThorlabsTLCamera(serial=serial)

    cam.set_exposure(exposure_time)
    cam.start_acquisition()  # start acquisition (automatically sets it up as well)

    image_data = shared_image_data.get_numpy_handle()
    cam.wait_for_frame()

    print(Text_Color.GREEN + f"Camera capturing for camera {str(serial)} started" + Text_Color.END)
    try:
        while stop_event.is_set() is False:
            cam.wait_for_frame()  # wait for the next available frame
            img = cam.read_newest_image()
            shared_image_data.get_lock().acquire()
            image_data[:, :, frame_counter.value % 25] = img
            frame_counter.value += 1
            if frame_counter.value == 25:
                process_start_switch.set()
            frame_ready_event.set()
            shared_image_data.get_lock().release()

    except (Exception, TimeoutError, KeyboardInterrupt) as ex:
        # stop_event.set()
        cam.stop_acquisition()
        print(f"Camera capturing for camera {str(serial)} stopped\n" f" -> Error: {repr(ex)}\n")


def local_gradients_calculation_xy(
    shared_image_array: MemorySharedNumpyArray,
    ROIs_xy: list,
    shared_array_x: MemorySharedNumpyArray,
    shared_array_y: MemorySharedNumpyArray,
    frame_counter: mp.Value,
    data_saving_x: mp.Queue,
    data_saving_y: mp.Queue,
    process_start_switch: mp.Event,
    frame_ready_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """
    Calculate the local gradients for the XY camera
    @param shared_image_array: Shared memory array for the image data
    @param ROIs_xy: 'Region of Interests' for the XY camera
    @param shared_array_x: Shared memory array for the X axis position data
    @param shared_array_y: Shared memory array for the Y axis position data
    @param frame_counter: Frame counter for the XY camera
    @param data_saving_x: Saving queue for the X axis particle position data
    @param data_saving_y: Saving queue for the Y axis particle position data
    @param process_start_switch: Process start switch for the XY camera
    @param frame_ready_event: Frame ready event for the XY camera
    @param stop_event: Stop event for the XY camera

    @return: None
    """
    import numpy as np
    import multiprocessing
    from time import perf_counter
    from datetime import datetime

    from system_variables import nPoints, pixel_to_nm_conv, weights_for_average, array_shift_2d

    internal_counter_xy = 0
    weight_array = weights_for_average(scaling_factor=2, num_points=25)

    print("Position Calculation for XY Camera Started")
    zero_time_xy = start_time_xy = perf_counter()

    print(f"Number of processors available: {multiprocessing.cpu_count()}\n")

    cores_to_use = min(
        len(ROIs_xy), mp.cpu_count()
    )  # Optimal number of cores to use is equal to the operations needed to be performed in parallel
    # otherwise the parallel processing becomes slower

    print(
        f"Number of cores (Deamonic child processes) to use: {Text_Color.BOLD}{Text_Color.DARKCYAN}{cores_to_use} (Physical + Logical){Text_Color.END}\n"
    )

    try:
        with multiprocessing.Pool(cores_to_use) as pool:
            while stop_event.is_set() is False:
                if process_start_switch.is_set() and frame_ready_event.is_set():
                    xy_image_array = shared_image_array.get_numpy_handle()
                    shared_image_array.get_lock().acquire()
                    image_buffer = xy_image_array.copy()
                    frame_value = frame_counter.value
                    shared_image_array.get_lock().release()

                    einsum_weights = np.roll(weight_array, frame_value % 25)
                    final_image = np.einsum("xyn, n -> xy", image_buffer, einsum_weights, optimize=True)

                    """img_crop = img_raw[y1:y1 + y2, x1:x1 + x2]"""
                    img_roi_lst = [final_image[ROI[1] : ROI[3] + ROI[1], ROI[0] : ROI[2] + ROI[0]] for ROI in ROIs_xy]

                    if internal_counter_xy == 0:
                        xy_coord_new = xy_coord_0 = np.array(pool.map(position_calc, img_roi_lst))
                    else:
                        xy_coord_new = np.array(pool.map(position_calc, img_roi_lst))

                    array_x = shared_array_x.get_numpy_handle()
                    array_y = shared_array_y.get_numpy_handle()
                    shared_array_x.get_lock().acquire()
                    shared_array_y.get_lock().acquire()

                    buffer_array_x = array_x.copy()
                    buffer_array_y = array_y.copy()

                    shared_array_x.get_lock().release()
                    shared_array_y.get_lock().release()

                    if internal_counter_xy >= nPoints:
                        buffer_array_x = array_shift_2d(buffer_array_x, -1, (xy_coord_new[:, 1] - xy_coord_0[:, 1]) * pixel_to_nm_conv)
                        buffer_array_y = array_shift_2d(buffer_array_y, -1, (xy_coord_new[:, 0] - xy_coord_0[:, 0]) * pixel_to_nm_conv)
                    else:
                        buffer_array_x[internal_counter_xy, :] = (xy_coord_new[:, 1] - xy_coord_0[:, 1]) * pixel_to_nm_conv
                        buffer_array_y[internal_counter_xy, :] = (xy_coord_new[:, 0] - xy_coord_0[:, 0]) * pixel_to_nm_conv

                    dt = datetime.now().strftime("%H_%M_%S_%f")
                    if internal_counter_xy >= nPoints:
                        data_saving_x.put([dt, *buffer_array_x[-1]])
                        data_saving_y.put([dt, *buffer_array_y[-1]])
                    else:
                        data_saving_x.put([dt, *buffer_array_x[internal_counter_xy]])
                        data_saving_y.put([dt, *buffer_array_y[internal_counter_xy]])

                    array_x = shared_array_x.get_numpy_handle()
                    array_y = shared_array_y.get_numpy_handle()
                    shared_array_x.get_lock().acquire()
                    shared_array_y.get_lock().acquire()

                    array_x[:] = buffer_array_x
                    array_y[:] = buffer_array_y
                    frame_ready_event.clear()

                    shared_array_x.get_lock().release()
                    shared_array_y.get_lock().release()

                    internal_counter_xy += 1

                if perf_counter() - zero_time_xy > 30:
                    print(
                        f"Average XY calculation rate = {Text_Color.YELLOW}{internal_counter_xy / (perf_counter() - start_time_xy):.2f}{Text_Color.END} FPS"
                    )
                    zero_time_xy = perf_counter()

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        multiprocessing.Pool(cores_to_use).close()
        print("\nCores processes closed for xy position calculation mapping")
        print(f"Error: {repr(ex)} -> Position Calculation for XY Camera Stopped\n")


def calculate_astigmatism_for_z(
    shared_image_array, shared_array_z, frame_count, data_saving_z, process_start_switch, z_frame_ready_event, stop_event
) -> None:

    import numpy as np
    from datetime import datetime
    from time import perf_counter

    from system_variables import calculate_astigmatism, array_shift, weights_for_average, nPoints

    internal_counter_z = 0
    weight_array = weights_for_average(scaling_factor=2, num_points=25)
    print("Astigmatism Calculation for Z Camera Started")

    zero_time_z = start_time_z = perf_counter()

    try:
        while stop_event.is_set() is False:
            if process_start_switch.is_set() and z_frame_ready_event.is_set():
                z_image_array = shared_image_array.get_numpy_handle()
                shared_image_array.get_lock().acquire()
                image_buffer = z_image_array.copy()
                frame_value = frame_count.value
                shared_image_array.get_lock().release()

                einsum_weights = np.roll(weight_array, frame_value % 25)
                final_image = np.einsum("xyn, n -> xy", image_buffer, einsum_weights, optimize=True)

                astigmatism = calculate_astigmatism(final_image, fft_pixels_in_x=50, fft_pixels_in_y=9)

                array_z = shared_array_z.get_numpy_handle()
                shared_array_z.get_lock().acquire()
                buffer_array_z = array_z.copy()
                shared_array_z.get_lock().release()

                if internal_counter_z == 0:
                    astigmatism_0 = buffer_array_z[0, 0] = astigmatism
                    data_saving_z.put([datetime.now().strftime("%H_%M_%S_%f"), astigmatism])

                elif 0 < internal_counter_z < nPoints:
                    buffer_array_z[internal_counter_z, 0] = astigmatism - astigmatism_0
                    data_saving_z.put([datetime.now().strftime("%H_%M_%S_%f"), astigmatism - astigmatism_0])

                else:
                    buffer_array_z[:, 0] = array_shift(buffer_array_z[:, 0], -1, astigmatism - astigmatism_0)
                    data_saving_z.put([datetime.now().strftime("%H_%M_%S_%f"), buffer_array_z[-1, 0]])

                array_z = shared_array_z.get_numpy_handle()
                shared_array_z.get_lock().acquire()
                array_z[:] = buffer_array_z
                z_frame_ready_event.clear()
                shared_array_z.get_lock().release()
                internal_counter_z += 1

            if perf_counter() - zero_time_z > 30:
                print(
                    f"Average Z calculation rate = {Text_Color.BLUE}{internal_counter_z / (perf_counter() - start_time_z):.2f}{Text_Color.END} FPS"
                )
                zero_time_z = perf_counter()

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print("Astigmatism Calculation Stopped\n" f"Error -> {repr(ex)}\n")


def stage_update(
    shared_array_x,
    shared_array_y,
    shared_array_z,
    x_position_start,
    y_position_start,
    z_position_start,
    data_saving_x,
    data_saving_y,
    data_saving_z,
    calibration_value,
    process_start_switch_xy,
    process_start_switch_z,
    stop_event,
) -> None:

    import numpy as np
    from time import sleep
    from datetime import datetime

    from pi_stage_controller import send_axis_position
    from system_variables import movement_range, inertia_correction_factor

    print("Stage Update Started")
    try:
        while stop_event.is_set() is False:
            if process_start_switch_xy.is_set() and process_start_switch_z.is_set():
                array_x = shared_array_x.get_numpy_handle()
                array_y = shared_array_y.get_numpy_handle()
                array_z = shared_array_z.get_numpy_handle()

                shared_array_x.get_lock().acquire()
                shared_array_y.get_lock().acquire()
                shared_array_z.get_lock().acquire()

                buffer_array_x = array_x.copy()
                buffer_array_y = array_y.copy()
                buffer_array_z = array_z.copy()

                shared_array_x.get_lock().release()
                shared_array_y.get_lock().release()
                shared_array_z.get_lock().release()

                buffer_array_x = np.mean(buffer_array_x, axis=1)
                buffer_array_y = np.mean(buffer_array_y, axis=1)

                dt = datetime.now().strftime("%H_%M_%S_%f")

                if abs(buffer_array_x[-1]) > movement_range and not np.isnan(buffer_array_x).any():
                    new_x_position = x_position_start - (buffer_array_x[-1] / inertia_correction_factor)
                    send_axis_position(0, new_x_position)
                    x_position_start = new_x_position
                    data_saving_x.put([dt, new_x_position])

                if abs(buffer_array_y[-1]) > movement_range and not np.isnan(buffer_array_y).any():
                    new_y_position = y_position_start - (buffer_array_y[-1] / inertia_correction_factor)
                    send_axis_position(1, new_y_position)
                    y_position_start = new_y_position
                    data_saving_y.put([dt, new_y_position])

                if abs(buffer_array_z[-1, 0]) > calibration_value * movement_range and not np.isnan(buffer_array_z).any():
                    new_z_position = z_position_start - ((buffer_array_z[-1, 0] / calibration_value) / inertia_correction_factor)
                    send_axis_position(2, new_z_position)
                    z_position_start = new_z_position
                    data_saving_z.put([dt, new_z_position])

            sleep(0.05)

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print(" Stage Update Stopped\n" f"Error -> {repr(ex)}\n")


def plotting_update(
    shared_array_x,
    shared_array_y,
    shared_array_z,
    calibration_value,
    update_time,
    process_start_switch_xy,
    process_start_switch_z,
    stop_event,
) -> None:
    """
    Plotting Parameters

    Making a plot gird for plotting the XY and Z coordinates
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from system_variables import nPoints

    print("Plotting Started")

    # Plotting the initial plots with empty arrays
    fig = plt.figure(figsize=(12, 8), num="Fluctuations in Particle Position", clear=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=(3, 1), height_ratios=(1, 1, 1), wspace=0.04, hspace=0.05)

    # X Coordinate Plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.tick_params(axis="x", direction="in", labelbottom=False, bottom=True)
    ax_hist.tick_params(axis="y", labelleft=False, left=True)

    # Y Coordinate Plots
    ay1 = fig.add_subplot(gs[1, 0])
    ay_hist = fig.add_subplot(gs[1, 1])
    ay_hist.tick_params(axis="x", direction="in", labelbottom=False, bottom=True)
    ay_hist.tick_params(axis="y", labelleft=False, left=True)

    # Z Coordinate Plots
    az1 = fig.add_subplot(gs[2, 0])
    az_hist = fig.add_subplot(gs[2, 1])
    az_hist.tick_params(axis="x", direction="in")
    az_hist.tick_params(axis="y", labelleft=False, left=True)

    ax1.set_xlim(0, nPoints)
    ax1.set_title("Fluctuations (per axis)"), ax1.set_ylabel("X nm Fluctuations")
    ax1.tick_params(axis="x", direction="in", labelbottom=False, bottom=True)

    ay1.set_xlim(0, nPoints)
    ay1.set_ylabel("Y nm Fluctuations")
    ay1.tick_params(axis="x", direction="in", labelbottom=False, bottom=True)

    az1.set_xlim(0, nPoints)
    az1.set_xlabel("Frame Number")
    az1.set_ylabel("Z nm Fluctuations")
    az1.tick_params(axis="x", direction="in")

    x_line = np.linspace(0, nPoints, nPoints)
    plotting_array = np.zeros(nPoints)

    # plots not redrawn every time but updated in-place
    (x_plot,) = ax1.plot(x_line, plotting_array, color="red")
    (y_plot,) = ay1.plot(x_line, plotting_array, color="green")
    (z_plot,) = az1.plot(x_line[:-1], plotting_array[1:] * (1 / calibration_value), color="blue")

    hist, bins = np.histogram(np.random.normal(0, 1, 300), bins=25, density=True)
    (x_hist,) = ax_hist.plot(hist, bins[:-1], color="red", drawstyle="steps", label="X")
    (y_hist,) = ay_hist.plot(hist, bins[:-1], color="green", drawstyle="steps", label="Y")
    (z_hist,) = az_hist.plot(hist, bins[:-1], color="blue", drawstyle="steps", label="Z")

    try:
        while stop_event.is_set() is False:
            if process_start_switch_xy.is_set() and process_start_switch_z.is_set():
                array_x = shared_array_x.get_numpy_handle()
                array_y = shared_array_y.get_numpy_handle()
                array_z = shared_array_z.get_numpy_handle()

                # reading the shared memory arrays into buffer for plotting
                shared_array_x.get_lock().acquire()
                shared_array_y.get_lock().acquire()
                shared_array_z.get_lock().acquire()

                buffer_array_x = array_x.copy()
                buffer_array_y = array_y.copy()
                buffer_array_z = array_z.copy()

                shared_array_x.get_lock().release()
                shared_array_y.get_lock().release()
                shared_array_z.get_lock().release()

                buffer_array_x = np.mean(buffer_array_x, axis=1)
                buffer_array_y = np.mean(buffer_array_y, axis=1)

                x_plot.set_ydata(buffer_array_x)
                y_plot.set_ydata(buffer_array_y)
                ax1.relim(visible_only=True)
                ax1.autoscale_view(True, False, True)
                ay1.relim(visible_only=True)
                ay1.autoscale_view(True, False, True)

                z_plot.set_ydata(buffer_array_z[1:] * (1 / calibration_value))
                az1.relim(visible_only=True)
                az1.autoscale_view(True, False, True)

                if not np.isnan(buffer_array_x).any() and not np.isnan(buffer_array_y).any() and not np.isnan(buffer_array_z).any():
                    hist_x, bins_x = np.histogram(buffer_array_x, bins=25, density=True)
                    hist_y, bins_y = np.histogram(buffer_array_y, bins=25, density=True)
                    hist_z, bins_z = np.histogram(buffer_array_z[1:] * (1 / calibration_value), bins=25, density=True)

                    x_hist.set_ydata(bins_x[:-1])
                    x_hist.set_xdata(hist_x)
                    ax_hist.relim(visible_only=True)
                    ax_hist.autoscale_view(True, True, True)
                    x_hist.set_label(f"std. = {np.std(buffer_array_x):.3f} nm")
                    ax_hist.legend(loc="upper right")

                    y_hist.set_ydata(bins_y[:-1])
                    y_hist.set_xdata(hist_y)
                    ay_hist.relim(visible_only=True)
                    ay_hist.autoscale_view(True, True, True)
                    y_hist.set_label(f"std. = {np.std(buffer_array_y):.3f} nm")
                    ay_hist.legend(loc="upper right")

                    z_hist.set_ydata(bins_z[:-1])
                    z_hist.set_xdata(hist_z)
                    az_hist.relim(visible_only=True)
                    az_hist.autoscale_view(True, True, True)
                    z_hist.set_label(f"std. = {np.std(buffer_array_z[1:] * (1 / calibration_value)):.3f} nm")
                    az_hist.legend(loc="upper right")

                plt.pause(update_time)

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print("Plotting Stopped\n" f"Error -> {repr(ex)}\n")


def save_particle_position_data(position_data_x, position_data_y, position_data_z, num_particles_xy, path_for_csv, stop_event) -> None:

    import numpy as np
    import pandas as pd
    from system_variables import dump_points, save_to_csv

    dump_x_position = pd.DataFrame(np.zeros((dump_points, num_particles_xy + 1)))
    dump_y_position = pd.DataFrame(np.zeros((dump_points, num_particles_xy + 1)))
    dump_z_position = pd.DataFrame(np.zeros((dump_points, 2)))

    count_x = 0
    count_y = 0
    count_z = 0

    try:
        while stop_event.is_set() is False:
            if not position_data_x.empty():
                dump_x_position.iloc[count_x, :] = position_data_x.get()
                count_x += 1

            if not position_data_y.empty():
                dump_y_position.iloc[count_y, :] = position_data_y.get()
                count_y += 1

            if not position_data_z.empty():
                dump_z_position.iloc[count_z, :] = position_data_z.get()
                count_z += 1

            if count_x == dump_points:
                save_to_csv(filename=f"{path_for_csv}/X_coordinates.csv", data=dump_x_position, mode="a")
                count_x = 0
                dump_x_position.iloc[:, :] = np.nan

            if count_y == dump_points:
                save_to_csv(filename=f"{path_for_csv}/Y_coordinates.csv", data=dump_y_position, mode="a")
                count_y = 0
                dump_y_position.iloc[:, :] = np.nan

            if count_z == dump_points:
                save_to_csv(filename=f"{path_for_csv}/Z_coordinates.csv", data=dump_z_position, mode="a")
                count_z = 0
                dump_z_position.iloc[:, :] = np.nan

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print("Data Saving for coordinates Stopped\n" f"Error -> {repr(ex)}")

    finally:
        save_to_csv(filename=f"{path_for_csv}/X_coordinates.csv", data=dump_x_position.dropna(), mode="a")
        save_to_csv(filename=f"{path_for_csv}/Y_coordinates.csv", data=dump_y_position.dropna(), mode="a")
        save_to_csv(filename=f"{path_for_csv}/Z_coordinates.csv", data=dump_z_position.dropna(), mode="a")
        print(f"Extra Data Saved for Particle positions\n")


def save_stage_position_data(position_data_x, position_data_y, position_data_z, path_for_csv, stop_event) -> None:

    import numpy as np
    import pandas as pd
    from system_variables import dump_points, save_to_csv

    dump_x_position = pd.DataFrame(np.zeros((dump_points, 2)))
    dump_y_position = pd.DataFrame(np.zeros((dump_points, 2)))
    dump_z_position = pd.DataFrame(np.zeros((dump_points, 2)))

    count_x = 0
    count_y = 0
    count_z = 0

    try:
        while stop_event.is_set() is False:

            if not position_data_x.empty():
                dump_x_position.iloc[count_x, :] = position_data_x.get()
                count_x += 1

            if not position_data_y.empty():
                dump_y_position.iloc[count_y, :] = position_data_y.get()
                count_y += 1

            if not position_data_z.empty():
                dump_z_position.iloc[count_z, :] = position_data_z.get()
                count_z += 1

            if count_x == dump_points:
                save_to_csv(filename=f"{path_for_csv}/X_position.csv", data=dump_x_position, mode="a")
                count_x = 0
                dump_x_position.iloc[:, :] = np.nan

            if count_y == dump_points:
                save_to_csv(filename=f"{path_for_csv}/Y_position.csv", data=dump_y_position, mode="a")
                count_y = 0
                dump_y_position.iloc[:, :] = np.nan

            if count_z == dump_points:
                save_to_csv(filename=f"{path_for_csv}/Z_position.csv", data=dump_z_position, mode="a")
                count_z = 0
                dump_z_position.iloc[:, :] = np.nan

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print(f"Data Saving for stage position Stopped\n" f"Error -> {repr(ex)}")

    finally:
        save_to_csv(filename=f"{path_for_csv}/X_position.csv", data=dump_x_position.dropna(), mode="a")
        save_to_csv(filename=f"{path_for_csv}/Y_position.csv", data=dump_y_position.dropna(), mode="a")
        save_to_csv(filename=f"{path_for_csv}/Z_position.csv", data=dump_z_position.dropna(), mode="a")
        print(f"Extra Data Saved for stage positions\n")


def save_camera_images(shared_image_array_xy, shared_image_array_z, path_for_images, delay_between_saves, stop_event) -> None:
    """
    Save the images from the XY and Z cameras to the specified path

    @param shared_image_array_xy: Shared memory array for the XY camera images
    @param shared_image_array_z: Shared memory array for the Z camera images
    @param path_for_images: Path for saving the images
    @param delay_between_saves: Delay between saving the images (in seconds)
    @param stop_event: Stop event for the process

    @return: None
    """
    from time import perf_counter
    from datetime import datetime
    import matplotlib.pyplot as plt

    zero_time = perf_counter()

    try:
        while stop_event.is_set() is False:
            if (perf_counter() - zero_time) > delay_between_saves:
                array_xy = shared_image_array_xy.get_numpy_handle()
                array_z = shared_image_array_z.get_numpy_handle()

                shared_image_array_xy.get_lock().acquire()
                shared_image_array_z.get_lock().acquire()
                buffer_array_xy = array_xy.copy()
                buffer_array_z = array_z.copy()
                shared_image_array_xy.get_lock().release()
                shared_image_array_z.get_lock().release()

                dt = datetime.now().strftime("%H_%M_%S_%f")

                plt.imsave(f"{path_for_images}/xy_image_{dt}.tiff", arr=buffer_array_xy[:, :, -1], cmap="gray")

                plt.imsave(f"{path_for_images}/z_image_{dt}.tiff", arr=buffer_array_z[:, :, -1], cmap="gray")

                zero_time = perf_counter()

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print(f"Image Saving Stopped\n" f"Error -> {repr(ex)}\n")


def frame_rate_calculation(frame_counter_xy, frame_counter_z, process_start_switch_xy, process_start_switch_z, stop_event) -> None:

    from time import perf_counter

    zero_time = start_time = perf_counter()

    try:
        while stop_event.is_set() is False:
            if process_start_switch_xy.is_set() and process_start_switch_z.is_set() and (perf_counter() - zero_time) > 30:
                print(
                    f"\nFrame Rate XY = {Text_Color.YELLOW}{frame_counter_xy.value / (perf_counter() - start_time):.2f}{Text_Color.END} FPS"
                    f" | Frame Rate Z = {Text_Color.BLUE}{frame_counter_z.value / (perf_counter() - start_time):.2f}{Text_Color.END} FPS"
                    f" | Frame Count XY = {Text_Color.YELLOW}{frame_counter_xy.value}{Text_Color.END}"
                    f" | Frame Count Z = {Text_Color.BLUE}{frame_counter_z.value}{Text_Color.END}"
                    f" | Time = {Text_Color.CYAN}{perf_counter() - start_time:.2f}{Text_Color.END} sec"
                )
                zero_time = perf_counter()

    except (Exception, KeyboardInterrupt) as ex:
        # stop_event.set()
        print(f"Frame Rate Calculation Stopped\n" f"Error -> {repr(ex)}\n")
