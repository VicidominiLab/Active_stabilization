from time import sleep
from functools import wraps
from typing import Callable, Any
from pipython import GCSDevice, pitools


def retry(retries: int = 3, delay: float = 1) -> Callable:
    """
    Attempt to call a function, if it fails, try again with a specified delay.

    :param retries: The max amount of retries you want for the function call
    :param delay: The delay (in seconds) between each function retry
    :return:
    """

    # Don't let the user use this decorator if they are high
    if retries < 1 or delay <= 0:
        raise ValueError("Are you high, mate?")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for i in range(1, retries + 1):  # 1 to retries + 1 since upper bound is exclusive

                try:
                    print(f"Running ({i}): {func.__name__}()")
                    return func(*args, **kwargs)
                except Exception as e:
                    # Break out of the loop if the max amount of retries is exceeded
                    if i == retries:
                        print(f"Error: {repr(e)}.")
                        print(f'"{func.__name__}()" failed after {retries} retries.')
                        break
                    else:
                        print(f"Error: {repr(e)} -> Retrying Connection...")
                        sleep(delay)  # Add a delay before running the next iteration

        return wrapper

    return decorator


Controller = "E-727"
STAGES = ["P-545.3R8H"]
serial = "0122079768"


@retry(retries=10, delay=3)
def connect_to_stage_controller() -> GCSDevice:
    """
    Connect to the stage controller
    """
    try:
        pidevice = GCSDevice(Controller)
        pidevice.ConnectUSB(serialnum=serial)
        pitools.startup(pidevice, stages=STAGES)
        print(f"Connected to {Controller} with serial number {serial} successfully\n")

    except Exception:
        raise ConnectionError(
            """Please check the connection to the stage controller and try again!!!"""
        )

    return pidevice


pidevice = connect_to_stage_controller()


def read_axis_position() -> list:
    """
    :return: position of the stage (all axes) in nm
    """
    # if testing:
    #     with GCSDevice(Controller) as pidevice:
    #         pidevice.ConnectUSB(serialnum='0122079768')
    #
    #     return pidevice.qPOS()
    #
    # else:
    #     pidevice = GCSDevice(Controller)
    #     pidevice.ConnectUSB(serialnum='0122079768')
    #     pitools.startup(pidevice, stages=STAGES)
    return pidevice.qPOS()


def send_axis_position(axis: int, position: float) -> None:
    """
    :param axis: axis number [1, 2, 3] = [X, Y, Z]
    :param position: position of the stage in nm
    """

    pidevice.MOV(pidevice.axes[axis], position)
    pitools.waitontarget(pidevice, axes=pidevice.axes[axis], polldelay=0.005)
