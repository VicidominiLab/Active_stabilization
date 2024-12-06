"""
Main script for the stabilization process using multiprocessing

Author: Sanket Patil, Eli Slenders, 2024
"""


if __name__ == "__main__":

    from stabilization_process_functions import (
        camera_capture,
        local_gradients_calculation_xy,
        calculate_astigmatism_for_z,
        stage_update,
        plotting_update,
        save_particle_position_data,
        save_stage_position_data,
        save_camera_images,
        frame_rate_calculation,
        Text_Color,
    )

    # Standard Library Imports
    import os
    import psutil
    import atexit
    import warnings
    import numpy as np
    import multiprocessing as mp
    from pipython import GCSDevice
    from time import perf_counter, sleep
    from pylablib.devices import Thorlabs

    # Local Imports
    from mp_shared_array import MemorySharedNumpyArray
    from pi_stage_controller import send_axis_position, read_axis_position, pidevice
    from system_variables import get_path_for_csv, get_path_for_images, load_config, save_config

    # boolean and fixed variables
    from system_variables import nPoints, plotting, data_saving, save_images, closed_loop_feedback

    from system_variables import (
        z_camera_serial,
        xy_camera_serial,
        get_exposure_time,
        Z_image_roi_for_FFT,
        get_rois_from_cam_image,
        get_fft_calibration_value,
        roi_select_for_acquisition,
    )

    # Check if the program is running on a Windows machine
    if os.name == "nt":
        print("Running on Windows -- Performing Windows Specific Checks\n")

        parent_pid = os.getppid()
        if psutil.Process(parent_pid).name() == "powershell.exe":
            print(Text_Color.GREEN + "Running from PowerShell - Functionality of code will be as expected\n" + Text_Color.END)
        else:
            warnings.warn(
                "Not running from PowerShell - Functionality of code may be affected\n We highly recommend running the code from PowerShell",
                FutureWarning,
            )

    else:
        warnings.warn("Not running on Windows - Functionality of code may/may not be affected depending on the terminal\n", FutureWarning)

    try:
        # ask the user if they want to load everything from config file or start from scratch
        load_from_config = input(
            Text_Color.BOLD + Text_Color.DARKCYAN + "\nDo you want to load the configuration from the config file? (y/n): " + Text_Color.END
        )

        stabilization_stop_event = mp.Event()
        process_start_switch_xy = mp.Event()
        process_start_switch_z = mp.Event()

        xy_frame_ready_event = mp.Event()
        z_frame_ready_event = mp.Event()

        path_for_saving_csv = get_path_for_csv()
        path_for_saving_images = get_path_for_images()

        """
        Pre stabilization initialization and Calibration
        Getting position reading from PI Stage Controller
    
        Starting position of the stage to stabilize
        """

        position = read_axis_position()

        x_position = x_position_og = list(position.values())[0]
        y_position = y_position_og = list(position.values())[1]
        z_position = z_position_og = list(position.values())[2]
        print(
            f"Stage Position --> X Position = {Text_Color.PURPLE}{x_position}{Text_Color.END}\n"
            f"Y Position = {Text_Color.PURPLE}{y_position}{Text_Color.END}\n"
            f"and Z Position --> {Text_Color.PURPLE}{z_position}{Text_Color.END}\n"
        )

        if load_from_config.lower()[0] == "y":

            """
            Load the configuration from the config file of the previous session
            Values loaded are:
            1. Exposure Time for XY Camera
            2. Exposure Time for Z Camera
            3. ROI for XY Camera
            4. ROI for Z Camera
            5. ROIs for tracking (Positions)
            6. FFT Calibration value per nm
            """

            print(Text_Color.DARKCYAN + "Loading Configuration from the configration file of the previous session" + Text_Color.END)
            print("\n")
            config_dict = load_config("config_dict.pkl")

            exposure_time_xy = config_dict["exposure_time_xy"]
            exposure_time_z = config_dict["exposure_time_z"]

            xy_camera_details = config_dict["xy_camera_details"]
            z_camera_details = config_dict["z_camera_details"]

            image_roi_xy = xy_camera_details[0]
            image_roi_Z = z_camera_details[0]

            ROIs_xy = config_dict["ROIs_xy"]

            fft_calib_value_per_nm = mp.Value("d", 0.0)
            fft_calib_value_per_nm.value = config_dict["fft_calib_value_per_nm"]

            with Thorlabs.ThorlabsTLCamera(xy_camera_serial) as cam:
                cam.set_roi(*xy_camera_details[1])

            with Thorlabs.ThorlabsTLCamera(z_camera_serial) as cam:
                cam.set_roi(*z_camera_details[1])

            print(Text_Color.BOLD + Text_Color.GREEN + "Loading Configuration Complete" + Text_Color.END)

        elif load_from_config.lower()[0] == "n":
            """
            Re-calibration of the system
            All values as mentioned above are recalibrated
            """

            # reading the exposure time of the cameras
            exposure_time_xy, exposure_time_z = get_exposure_time()

            """Select Region of Interests to be tracked"""
            xy_camera_details = roi_select_for_acquisition()  # Select ROIs for XY camera
            image_roi_xy = xy_camera_details[0]

            # center calculation and ROI changing for Z camera
            z_camera_details = Z_image_roi_for_FFT()  # Select ROI for Z camera
            image_roi_Z = z_camera_details[0]

            ROIs_xy = get_rois_from_cam_image(exposure_time_xy)

            # Z axis FFT calibration
            fft_calib_value_per_nm = mp.Value("d", 0.0)
            fft_calib_value_per_nm.value = min(
                0.001,
                get_fft_calibration_value(z_position=z_position, exposure_time_z=exposure_time_z, calibration_z_nm=50, num_z_points=5),
            )

            # Save the configuration for the current session in a .pkl file
            config_dict: dict = {
                "exposure_time_xy": exposure_time_xy,
                "exposure_time_z": exposure_time_z,
                "xy_camera_details": xy_camera_details,
                "z_camera_details": z_camera_details,
                "ROIs_xy": ROIs_xy,
                "fft_calib_value_per_nm": fft_calib_value_per_nm.value,
            }

        save_config(config_dict)
        print("Configration Saved for current session")
        print("\n")

        # Setting all axes to original position after calibration (assuming the sample has drifted a bit)
        send_axis_position(0, x_position)
        send_axis_position(1, y_position)
        send_axis_position(2, z_position)
        print("Stage moved to original position")
        print("\n")

        pidevice.unload()

        print(f"Z axis FFT Calibration Complete. FFT value = {Text_Color.YELLOW}{fft_calib_value_per_nm.value:.6f} {Text_Color.END}per nm")
        print("\n")

        print(
            f"Exposure Time for XY Camera = {Text_Color.PURPLE}{exposure_time_xy * 1000:.4f}{Text_Color.END} ms "
            f"and Z Camera = {Text_Color.PURPLE}{exposure_time_z * 1000:.4f}{Text_Color.END} ms\n"
        )

        # Rearranging the rois to be compatible with the algorithm for tracking
        for roi in ROIs_xy:
            roi[0] = int(roi[0] - (roi[2] / 2))
            roi[1] = int(roi[1] - (roi[3] / 2))

        # position data arrays for each axis
        data_array_x = MemorySharedNumpyArray(shape=(nPoints, len(ROIs_xy)), dtype=np.float64, sampling=1)
        data_array_y = MemorySharedNumpyArray(shape=(nPoints, len(ROIs_xy)), dtype=np.float64, sampling=1)
        data_array_z = MemorySharedNumpyArray(shape=(nPoints, 1), dtype=np.float64, sampling=1)

        # fill the arrays with nan values
        # This is done to avoid problems with position average calculations
        data_array_x.get_numpy_handle().fill(np.nan)
        data_array_y.get_numpy_handle().fill(np.nan)
        data_array_z.get_numpy_handle().fill(np.nan)

        # Image buffer for both cameras
        image_array_xy = MemorySharedNumpyArray(shape=(int(image_roi_xy[1]), int(image_roi_xy[0]), 25), dtype=np.int16, sampling=1)
        image_array_z = MemorySharedNumpyArray(shape=(int(image_roi_Z[1]), int(image_roi_Z[0]), 25), dtype=np.int16, sampling=1)

        # fill the arrays with nan values
        # Similar reason as above
        image_array_xy.get_numpy_handle().fill(np.nan)
        image_array_z.get_numpy_handle().fill(np.nan)

        """
        Shared memory variables and data saving queues for the stabilization main processes
        """
        frame_counter_xy = mp.Value("i", 0)
        frame_counter_z = mp.Value("i", 0)

        data_for_saving_x_particle_position = mp.Queue()
        data_for_saving_y_particle_position = mp.Queue()
        data_for_saving_z_astigmatism = mp.Queue()

        data_for_saving_x_stage_position = mp.Queue()
        data_for_saving_y_stage_position = mp.Queue()
        data_for_saving_z_stage_position = mp.Queue()

        """
        Defining all the processes for the stabilization
        Processes are only defined here but not started
        All the parameters are passed to the processes as arguments at this stage
        """
        P1_xy_cam = mp.Process(
            target=camera_capture,
            args=(
                xy_camera_serial,
                exposure_time_xy,
                image_array_xy,
                frame_counter_xy,
                process_start_switch_xy,
                xy_frame_ready_event,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P2_z_cam = mp.Process(
            target=camera_capture,
            args=(
                z_camera_serial,
                exposure_time_z,
                image_array_z,
                frame_counter_z,
                process_start_switch_z,
                z_frame_ready_event,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P3_xy_calc = mp.Process(
            target=local_gradients_calculation_xy,
            args=(
                image_array_xy,
                ROIs_xy,
                data_array_x,
                data_array_y,
                frame_counter_xy,
                data_for_saving_x_particle_position,
                data_for_saving_y_particle_position,
                process_start_switch_xy,
                xy_frame_ready_event,
                stabilization_stop_event,
            ),
            daemon=False,  # Deamon is set to False because deamonic processes are not allowed to create child processes
            # Child-processes are created by the local_gradients_calculation_xy function for parallel processing (Line 143 and 145, stabilization_process_functions.py)
        )

        P4_z_calc = mp.Process(
            target=calculate_astigmatism_for_z,
            args=(
                image_array_z,
                data_array_z,
                frame_counter_z,
                data_for_saving_z_astigmatism,
                process_start_switch_z,
                z_frame_ready_event,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P5_stage_update = mp.Process(
            target=stage_update,
            args=(
                data_array_x,
                data_array_y,
                data_array_z,
                x_position,
                y_position,
                z_position,
                data_for_saving_x_stage_position,
                data_for_saving_y_stage_position,
                data_for_saving_z_stage_position,
                fft_calib_value_per_nm.value,
                process_start_switch_xy,
                process_start_switch_z,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P6_plotting = mp.Process(
            target=plotting_update,
            args=(
                data_array_x,
                data_array_y,
                data_array_z,
                fft_calib_value_per_nm.value,
                0.1,  # seconds
                process_start_switch_xy,
                process_start_switch_z,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P7_coordinate_saving = mp.Process(
            target=save_particle_position_data,
            args=(
                data_for_saving_x_particle_position,
                data_for_saving_y_particle_position,
                data_for_saving_z_astigmatism,
                len(ROIs_xy),
                path_for_saving_csv,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P8_position_saving = mp.Process(
            target=save_stage_position_data,
            args=(
                data_for_saving_x_stage_position,
                data_for_saving_y_stage_position,
                data_for_saving_z_stage_position,
                path_for_saving_csv,
                stabilization_stop_event,
            ),
            daemon=True,
        )

        P9_image_saving = mp.Process(
            target=save_camera_images,
            args=(image_array_xy, image_array_z, path_for_saving_images, 30, stabilization_stop_event),
            daemon=True,
        )

        P10_frame_rate = mp.Process(
            target=frame_rate_calculation,
            args=(frame_counter_xy, frame_counter_z, process_start_switch_xy, process_start_switch_z, stabilization_stop_event),
            daemon=True,
        )

        print("Starting all processes\n")

        """ 
        Starting the processes based on the user input
        Each process is started in a separate try-except block to handle errors in a different memory space with shared data
        """
        P1_xy_cam.start()
        P2_z_cam.start()
        P3_xy_calc.start()
        P4_z_calc.start()
        P5_stage_update.start() if closed_loop_feedback else None
        P6_plotting.start() if plotting else None
        P7_coordinate_saving.start() if data_saving else None
        P8_position_saving.start() if data_saving else None
        P9_image_saving.start() if save_images else None
        P10_frame_rate.start()

        # Print the process codes for each process (for per process analysis (memory, CPU, kernel etc.) and debugging)
        print(f"XY Camera Process code: {Text_Color.GREEN}{P1_xy_cam.pid}{Text_Color.END}")
        print(f"Z Camera Process code: {Text_Color.GREEN}{P2_z_cam.pid}{Text_Color.END}")
        print(f"XY Calculation Process code: {Text_Color.GREEN}{P3_xy_calc.pid}{Text_Color.END}")
        print(f"Z Calculation Process code: {Text_Color.GREEN}{P4_z_calc.pid}{Text_Color.END}")
        print(f"Stage Update Process code: {Text_Color.GREEN}{P5_stage_update.pid}{Text_Color.END}") if closed_loop_feedback else None
        print(f"Plotting Process code: {Text_Color.GREEN}{P6_plotting.pid}{Text_Color.END}") if plotting else None
        print(f"Coordinate Saving Process code: {Text_Color.GREEN}{P7_coordinate_saving.pid}{Text_Color.END}") if data_saving else None
        print(f"Position Saving Process code: {Text_Color.GREEN}{P8_position_saving.pid}{Text_Color.END}") if data_saving else None
        print(f"Image Saving Process code: {Text_Color.GREEN}{P9_image_saving.pid}{Text_Color.END}") if save_images else None
        print(f"Frame Rate Process code: {Text_Color.GREEN}{P10_frame_rate.pid}{Text_Color.END}")

        print("\nStarted!")

        start_time = perf_counter()

        """
        DO NOT DELETE THE FOLLOWING LINES
        This is to keep the main process running until the program is stopped
        If these lines are deleted, the program will stop immediately after starting
        Because the processes are not inherently persistent and the processes will stop after the last line of code
        """

        while stabilization_stop_event.is_set() is False:
            sleep(10)

    except (
        KeyboardInterrupt,
        TimeoutError,
        Exception,
    ) as e:  # Stop the program by pressing Ctrl+C at any time, Also stops the program in case of any error
        print(Text_Color.RED + "\nError {repr(e)}: -> Program Stopped at Main Process\n" + Text_Color.END)

        stabilization_stop_event.set()
        print(Text_Color.RED + "Stopping all processes\n" + Text_Color.END)

        # Joining all processes for a graceful exit [Based on hope - there is no graceful kill in multiprocessing]
        P1_xy_cam.join()
        P2_z_cam.join()
        P3_xy_calc.join()
        P4_z_calc.join()
        P5_stage_update.join() if closed_loop_feedback else None
        P6_plotting.join() if plotting else None
        P7_coordinate_saving.join() if data_saving else None
        P8_position_saving.join() if data_saving else None
        P9_image_saving.join() if save_images else None
        P10_frame_rate.join()

    finally:

        """
        Exit handler
        Is executed whenever the program is stopped - Function call is not required
        """

        @atexit.register
        def exit_handler():

            # Hard termination of all processes - stops the stabilization processes
            print(Text_Color.RED + "Terminating all processes\n" + Text_Color.END)
            P1_xy_cam.terminate()
            P2_z_cam.terminate()
            P3_xy_calc.terminate()
            P4_z_calc.terminate()
            P5_stage_update.terminate() if closed_loop_feedback else None
            P6_plotting.terminate() if plotting else None
            P7_coordinate_saving.terminate() if data_saving else None
            P8_position_saving.terminate() if data_saving else None
            P9_image_saving.terminate() if save_images else None
            P10_frame_rate.terminate()

            final_time = perf_counter()

            # save the frame rate to a text file
            names = np.array(["XY_Frames", "Z_Frames", "Frame_Rate_XY", "Frame_Rate_Z", "Total_Time"])
            floats = np.array(
                [
                    frame_counter_xy.value,
                    frame_counter_z.value,
                    frame_counter_xy.value / (final_time - start_time),
                    frame_counter_z.value / (final_time - start_time),
                    final_time - start_time,
                ]
            )

            ab = np.zeros(names.size, dtype=[("var1", "U15"), ("var2", float)])
            ab["var1"] = names
            ab["var2"] = floats

            np.savetxt(f"{path_for_saving_csv}/Framerate_information.txt", ab, fmt="%10s %10.3f")

            print("Setting cameras back to default settings\n\n")
            with Thorlabs.ThorlabsTLCamera("09490") as cam:
                cam.set_roi(0, 1440, 0, 1080, 1, 1)

            with Thorlabs.ThorlabsTLCamera("22424") as cam:
                cam.set_roi(0, 1440, 0, 1080, 1, 1)

            with GCSDevice("E-727") as pi_stage:
                pi_stage.ConnectUSB(serialnum="0122079768")
                pos = pi_stage.qPOS()

                print(
                    f"Final position of the stage: X = {list(pos.values())[0]},  Y = {list(pos.values())[1]},  Z = {list(pos.values())[2]}"
                )
                print("")
                # print the difference between the initial and final position of the stage
                print(
                    f"Stage moved since start:\n"
                    f"X = {Text_Color.PURPLE}{(list(pos.values())[0] - x_position_og) * 1000:.4f}{Text_Color.END} nm\n"
                    f"Y = {Text_Color.PURPLE}{(list(pos.values())[1] - y_position_og) * 1000:.4f}{Text_Color.END} nm\n"
                    f"Z = {Text_Color.PURPLE}{(list(pos.values())[2] - z_position_og) * 1000:.4f}{Text_Color.END} nm\n"
                )

                pi_stage.CloseConnection()

            print("Stage connection closed")
            print("Program Ended")
