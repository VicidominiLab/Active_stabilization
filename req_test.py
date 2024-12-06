# This script is used to check if all the required packages are installed correctly and if the camera and stage are connected
import warnings

try:
    import matplotlib, numba, numpy, cv2, pandas, pipython, psutil, pylablib, sklearn, scipy, tqdm

    print("All required packages imported successfully")
except ImportError as e:
    print(f"Error importing {e} - Please check the package is installed correctly")


dict_req = {
    matplotlib: "3.9.2",
    numba: "0.60.0",
    numpy: "1.23.4",
    cv2: "4.10.0.84",
    pandas: "2.0.2",
    pipython: "2.10.2.1",
    psutil: "5.9.0",
    pylablib: "1.4.1",
    sklearn: "1.1.3",
    scipy: "1.14.1",
    tqdm: "4.64.1",
}

for key, value in dict_req.items():
    if key.__version__ == value:
        print(f"{key} is available in correct version")
    else:
        warnings.warn(f"{key} is not available in correct version - May cause unexpected behaviour", FutureWarning, stacklevel=2)


# check if the camera is connected
try:
    from pylablib.devices import Thorlabs

    print("Getting list of all connected cameras")
    camera_list = Thorlabs.TLCamera.list_cameras()

    if len(camera_list) == 0:
        print("No cameras detected")

    else:
        print("Getting device info for all detected cameras")
        for i in range(len(camera_list)):
            print(Thorlabs.ThorlabsTLCamera(serial=camera_list[i]).get_device_info())

except ImportError as e:
    print(f"Error: {e}. Error in importing or connecting cameras - please check and retry.")


# check if the stage is connected

try:
    from pipython import GCSDevice, pitools

    with GCSDevice() as pidevice:
        print("search for controllers...")
        devices = pidevice.EnumerateUSB()
        for i, device in enumerate(devices):
            print(f"{device} available")

        pidevice.ConnectUSB(devices[0])
        print("connected: {}".format(pidevice.qIDN().strip()))

except ImportError as e:
    print(f"Error: {e}. Error in importing or connecting stages - please check and retry.")


print("\n\nIf no errors are shown above, all requirements are installed correctly - You are good to go!!!")
