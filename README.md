# Active Stabilization

Active 3D Stabilization: Open source code for active 3D stabilization using multiprocessing

## Installation

The software does not require an installation process to function.

## Requirements
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

Additionally, the software required the installation of third-party software - **Thorcam** [(Download Link)](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) and **piMikroMove** to control the Thorlabs cameras and PI piezo stage. These software provide the necessary .dll files for Python to interface with the devices.

### Thorcam Installation
During the installation of Thorcam software, the user will be asked to choose the driver required for the cameras (Image 1)
1) ![image](https://github.com/user-attachments/assets/7adcc652-dea9-464e-85d2-d551166c8d85)


The user then should select the "_USB_" option and select "_Entire feature will be installed on local hard drive_" (Image 2)

2) ![image](https://github.com/user-attachments/assets/95e02cdc-3e70-4fee-ac45-14bbb77908b9)

Finally, the user should do the same for the "_Camera Link_" option (Image 3). After this, the user can proceed with the installation of the software.

3) ![image](https://github.com/user-attachments/assets/7e356a86-0457-4a4b-9e20-a0106d315264)

## Testing

The software is designed and tested using Thorlabs Zelux (CS165MU1/M) and PI piezo stage (P-545.3R8S PInano, E-727 Controller).
The software has been tested on two computers with configuration - CPU

	1) Intel i7-9750H - 2.60GHz, 16 GB RAM, NVIDIA GeForce RTX 2060 [Laptop]
    2) Intel Xeon W-2245 - 3.90GHz, 128GB RAM, NVIDIA Quadro P1000 [Workstation]

The software is able to achieve comparable performance on both machines.

## Instruction for running code locally
The software can be set and used in the following steps:
1) create a new Python environment
2) clone repository
3) install requirements and associated software
4) perform hardware checks
5) run the main code script

Details for each step are given below:
1) Create a new Python, preferably for compatibility reasons, using Python Venv or Conda (if using Anaconda)

	a) Pure Python Method
	- Install Python 3.10.X from the package installer [(Download Link)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
	- Make a new Python environment using the command :
- 		python -m venv active_stabilization

	b) Anaconda Method
	- If you are using Anaconda, you can make a new environment using the command:
- 		conda create --name active_stabilization python=3.10
 		conda activate active_stabilization

2) Once the environment is created, clone the active stabilization code on your local machine

	The repository can be cloned on a local system using the command (preferably in a new folder)
	
		git clone https://github.com/VicidominiLab/Active_stabilization.git
	 
3) After cloning, install the requirements as mentioned in "_requirements.txt_" file
	 All the requirements can be installed using the command:

		pip install -r requirements.txt

	After the requirements, Install the Thorcam and piMikroMove software as mentioned in [Requirements](https://github.com/VicidominiLab/Active_stabilization/edit/main/README.md#requirements)


4) After installing the requirements, run the "_req_test.py_" file to perform the checks for all the requirements using the command:

 		python req_test.py


5) Finally, if there are no errors, proceed to run the main stabilization script to start the active stabilization:

   		python stabilization_main.py


## Code Execution

The code can be executed on Windows in any terminal program (Powershell, console emulator, terminal). It has been developed on Windows and tested in Powershell - which we recommend running the code with.
The execution of the scripts is shown in the videos below

1) Execution **without** loading from the configuration file:- [Without_config.webm](https://github.com/user-attachments/assets/e62e7e1c-ca4e-402b-a874-8ce409bddd9c)

2) Execution **with** loading from the configuration file:- [with_config.webm](https://github.com/user-attachments/assets/b12233ac-b347-4cb7-a838-319190653644)

The button presses are shown on the left side of the video.



   


