<img src="https://yt3.ggpht.com/ytc/AKedOLS6CuwrrOvURWxJNMZt0KjWetOmkT6MJIP8DuGItQ=s900-c-k-c0x00ffffff-no-rj" align="right" width="150" height="150"/>

# SharkSight
SharkSight: Dive into the Depths of Computer Vision with Hammerhead Precision

## Introduction

SharkSight is a computer vision application designed to detect and track objects in real-time using the Jetson Nano platform. This application is specifically designed to detect and track cubes and cones for the 2023 FIRST Robotics Competition Game: Charged Up.

## Installation

### Prerequisites
The following components are needed to run SharkSight:

- NVIDIA Jetson module (Xavier, Nano, TX2, etc.)
- Camera (or cameras) compatible with the Jetson module
- Python 3.6 or later installed
- OpenCV 4.1.1 or later installed
- numpy 1.16.2 or later installed

### Installing Jetson Inference
SharkSight uses the Jetson Inference library to perform object detection and tracking. To install Jetson Inference, follow the instructions listed in the [Jetson-Inference GitHub repository](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md#building-the-project-from-source)

> Note: When installing the pretrained models, make sure to install the `ssd-mobilenet-v2` model.


### Custom Model Training
SharkSight uses a modified version of the `ssd-mobilenet-v2` model to detect and track cubes and cones. To train a custom model, follow the instructions listed in the [Jetson-Inference GitHub repository](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md#training-your-model)

Ensure that the `ssd-mobilenet-v2` model is installed before training a custom model. Additionally, make sure the dataset is in the Pascal VOC format. The dataset used to train the custom model can be found [here](https://universe.roboflow.com/robotics-qlzra/charged-up-cubes-and-cones-courtesy-of-frc-team-88).

The following commands were used to train the custom model:

```bash
python3 train_ssd.py --dataset-type=voc --data=data/cubes_cones --model-dir=/models/cubes_cones --batch-size=4 --workers=2
```
And Converting to ONNX format:
```bash
python3 onnx_export.py --model-dir=models/cubes_cones
```

#### Creating A Swap File (Optional if using pre-trained model)
If you are training a custom model, you will need to create a swap file to prevent the Jetson Nano from running out of memory. To create a swap file, follow the instructions:

1. Check if you have any existing swap space by running the command `sudo swapon --show`. If you see any output, you already have some swap space configured.
2. Run the command `sudo fallocate -l 4G /swapfile` to create a 4GB swap file.
3. Run the command `sudo chmod 600 /swapfile` to set the correct permissions for the swap file.
4. Set up the swap file by running the command `sudo mkswap /swapfile`.
5. Enable the swap file by running the command `sudo swapon /swapfile`.
6. To make the swap file permanent, run the command `sudo nano /etc/fstab` and add the following line to the end of the file: `/swapfile swap swap defaults 0 0`
7. To check if the swap file was created successfully, run the command `sudo swapon --show`.

### Installing PyNetworkTables
SharkSight uses PyNetworkTables to send data to the robot. WPILib 2023 requires Ubuntu 22.04 which is not supported by Jetson, so NT4 is not supported (`pyntcore`)

* Run `python3 -m pip install pynetworktables`

### Installing CScore
SharkSight uses CScore to stream video to the dashboard. To install CScore, run the following commands:


```
export CPPFLAGS=-I/usr/include/opencv4
python3 -m pip install robotpy-cscore===2022.0.3
```

### Creating Persistent USB Camera Connections
SharkSight uses USB cameras to detect and track objects. To ensure that the cameras are always connected to the same Jetson port, follow the instructions:

1. Run the command `lsusb` to list all USB devices connected to the Jetson.
2. Create a udew rule by running the command `sudo nano /etc/udev/rules.d/99-usb-serial.rules` and add the following line to the file: 
   `SUBSYSTEM=="video4linux", KERNELS=="2-2.4:1.0", NAME="video0"` and `SUBSYSTEM=="video4linux", KERNELS=="2-2.4:1.1", NAME="video1"` (replace `2-2.4:1.0` and `2-2.4:1.1` with the kernel names of your cameras)
3. Run the command `sudo udevadm control --reload-rules` to reload the rules.

### Setting a Static IP Address
SharkSight uses a static IP address to communicate with the robot. To set a static IP address, follow the instructions:

1. Open the `/etc/default/networking` file by running the command `sudo nano /etc/default/networking`.
2. Change CONFIGURE_INTERFACES to `no`. 
3. Open the `/etc/network/interfaces` file by running the command `sudo nano /etc/network/interfaces`.
```
source-directory /etc/network/interfaces.d
source interfaces.d/eth0 
```
4. Create a new file in the `/etc/network/interfaces.d` directory by running the command `sudo nano /etc/network/interfaces.d/eth0`.
```
    auth eth0
    iface eth0 inet static
    address 10.TE.AM.11
    netmask 255.255.255.0
    gateway 10.TE.AM.4
```
1. `sudo reboot`
 
### Running SharkSight
To run SharkSight, run the following command:

```
python3 shark_sight.py [--threshold THRESHOLD] [--capture-height CAPTURE_HEIGHT]
                      [--capture-width CAPTURE_WIDTH] [--stream-height STREAM_HEIGHT]
                      [--stream-width STREAM_WIDTH] [--stream-compression STREAM_COMPRESSION]
                      [--display]
```

Make sure to ensure that the cameras are connected to the Jetson before running SharkSight. Additionally, make sure that the robot is connected to the same network as the Jetson. In the code, ensure the team number is correct and that `net` variable is set to the correct model and directory.

The script will begin capturing and processing images from the connected camera(s), detecting and tracking objects in real-time. The application will output a live video stream to the CameraServer, which can be viewed using the SmartDashboard application

### Command Line Arguments
The following command line arguments can be used to configure SharkSight:

- `--threshold`: Sets the minimum confidence level for object detection (default is 0.5).
- `--capture-height`: Sets the resolution height to capture images from the camera (default is 720).
- `--capture-width`: Sets the resolution width to capture images from the camera (default is 1280).
- `--stream-height`: Sets the resolution to stream to the CameraServer (default is 270).
- `--stream-width`: Sets the resolution to stream to the CameraServer (default is 480).
- `--stream-compression`: Sets the compression to stream for clients that do not specify it (default is 30).
- `--display`: Enables the display output (default is false).

### Starting SharkSight on Boot
To start SharkSight on boot, follow the instructions:

1. Create a new service by running the command `sudo nano /etc/systemd/system/shark_sight.service`.
2. Add the following lines to the file:
```
    [Unit]
    Description=SharkSight
    After=network.target
    Wants=network-online.target systemd-networkd-wait-online.service

    [Service]
    Type=simple
    User=NAME
    ExecStart=/usr/bin/python3 /home/NAME/shark_sight/shark_sight.py
    Restart=on-failure
    RestartSec=1

    [Install]
    WantedBy=multi-user.target
```
1. Test the service by running the command `sudo systemctl start shark_sight.service` and `journalctl -u shark_sight.service`.
2. If the service starts successfully, enable it by running the command `sudo systemctl enable shark_sight.service`.

### Headless Mode
SharkSight can be run in headless mode, which means that it will not display any output to the screen. To run SharkSight in headless mode, follow the instructions:

1. Run the command `sudo systemctl set-default multi-user.target` to disable the graphical user interface.
   
### Using SSH
In order to change files without having to connect a monitor and keyboard to the Jetson, you can use SSH. To use SSH, follow the instructions:

`$ ssh -Y user@jetson-name.local` in the Command Prompt
