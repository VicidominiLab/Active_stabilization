# Active Stabilization

Active 3D Stabilization: Open source code for active 3D stabilization using multiprocessing

## Installation

The software does not require an installation process to function.

### Requirements
The software required prerequisites for functioning properly.
It requires the following Python packages

    matplotlib
    numba (required for JIT compiling numpy)
    numpy
    opencv_python
    pandas
    PIPython (PI piezo stage)
    psutil
    pylablib (Thorlabs camera control)
    scikit_learn
    scipy
    tqdm

All the specific versions of the packages are mentioned and can be installed from the "requirements.txt" file. All packages can be directly installed using the command 

    pip install -r requirements.txt 

Additionally, the software required the installation of third-party software - **Thorcam** and **piMikroMove** to control the Thorlabs cameras and PI piezo stage. These software provide the necessary .dll files for Python to interface with the devices.

## Testing

The software is designed and tested using Thorlabs Zelux (CS165MU1/M) and PI piezo stage (P-545.3R8S PInano, E-727 Controller).
The software has been tested on two computers with configuration - CPU

	1) Intel i7-9750H - 2.60GHz, 16 GB RAM, NVIDIA GeForce RTX 2060 [Laptop]
    2) Intel Xeon W-2245 - 3.90GHz, 128GB RAM, NVIDIA Quadro P1000 [Workstation]

The software is able to achieve comparable performance on both machines.







