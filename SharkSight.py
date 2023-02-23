import jetson.inference
import jetson.utils
from cscore import CameraServer
import cscore
from networktables import NetworkTables
import json
import time
import math
import logging
import argparse
import os
import cv2
import numpy as np

def drawCrossHairs(image, x, y, r, g, b, a, size, gapSize, thickness):
	jetson.utils.cudaDrawLine(image, (x, y - size // 2), (x, y - gapSize // 2), (r,g,b,a), thickness)
	jetson.utils.cudaDrawLine(image, (x, y + size // 2), (x, y + gapSize // 2), (r,g,b,a), thickness)

	jetson.utils.cudaDrawLine(image, (x - size // 2, y), (x - gapSize // 2, y), (r,g,b,a), thickness)
	jetson.utils.cudaDrawLine(image, (x + size // 2, y), (x + gapSize // 2, y), (r,g,b,a), thickness)

parser = argparse.ArgumentParser(description='Shark Sight')

parser.add_argument('--threshold', type=float, default=0.5, help='Confidence Minimum')
parser.add_argument('--capture-height', type=int, default=720,
                    help='The resolution height to capture images from the camera.')
parser.add_argument('--capture-width', type=int, default=1280,
                    help='The resolution width to capture images from the camera.')
parser.add_argument('--stream-height', type=int, default=270,
                    help='The resolution to stream to the CameraServer.')
parser.add_argument('--stream-width', type=int, default=480,
                    help='The resolution to stream to the CameraServer.')
parser.add_argument('--stream-compression', type=int, default=30,
                    help='The compression to stream.')
parser.add_argument('--display', '-d', action='store_true',
                    help='Default: False.')
args = parser.parse_args()

captureScale = args.capture_height / 720

captureArea = args.capture_height * args.capture_width
crosshairX = args.capture_width // 2
crosshairY = args.capture_height // 2

logging.basicConfig(level=logging.DEBUG)
cs = CameraServer.getInstance()
cs.enableLogging()

csSource = cscore.CvSource("Driver", cscore.VideoMode.PixelFormat.kMJPEG, args.stream_width, args.stream_height, 24)
server = cs.startAutomaticCapture(camera=csSource, return_server=True)
server.setCompression(args.stream_compression)

csSource2 = cscore.CvSource("Driver2", cscore.VideoMode.PixelFormat.kMJPEG, args.stream_width, args.stream_height, 24)
server2 = cs.startAutomaticCapture(camera=csSource2, return_server=True)
server2.setCompression(args.stream_compression)

NetworkTables.startClientTeam(226)

sd = NetworkTables.getTable("SharkSight")

net = jetson.inference.detectNet(argv=["--model=/home/hammerheads/jetson-inference/python/training/detection/ssd/models/cubes/ssd-mobilenet.onnx", "--labels=/home/hammerheads/jetson-inference/python/training/detection/ssd/models/cubes/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)

camera = jetson.utils.videoSource(
	"/dev/video0",
	argv=[
		f"--input-width={str(args.capture_width)}",
		f"--input-height={str(args.capture_height)}",
	],
)
camera2 = jetson.utils.videoSource(
	"/dev/video1",
	argv=[
		f"--input-width={str(args.capture_width)}",
		f"--input-height={str(args.capture_height)}",
	],
)

sd.putString("Intake Camera FPS", camera.GetFrameRate())
sd.putString("Gripper Camera FPS", camera2.GetFrameRate())

sd.putBoolean("Shutdown", False)
sd.putBoolean("Enabled", True)

display = jetson.utils.videoOutput("display://0") if args.display else None

smallImg = None
bgrSmallImg = None
smallImg2 = None
bgrSmallImg2 = None

startTime = time.time()

while True:
	if sd.getBoolean("Shutdown", False):
		os.system('systemctl poweroff')

	if (sd.getNumber("CPU Temp", 0) > 85 or sd.getNumber("GPU Temp", 0) > 85):
		time.sleep(.02)
		continue

	if not sd.getBoolean("Enabled", True):
		time.sleep(.02)
		continue

	sd.putString("Status", "Running")

	img = camera.Capture()
	img2 = camera2.Capture()

	detections = net.Detect(img, overlay='none')
	detections2 = net.Detect(img2, overlay='none')

	closestIntakeDetection = None
	closestIntakeDetectionDistance = 10000

	ntIntakeDetections = []
	for detection in detections:
		areaPercent = ((detection.Area / captureArea) * 100)
		targetX = detection.Center[0] - crosshairX
		targetY = detection.Center[1] - crosshairY
		targetDistance = math.sqrt(targetX**2 + targetY**2)

		className = "cubes"
		if (detection.ClassID == 2):
			className = "cones"
		ntDetection = {
			"ClassID": detection.ClassID,
			"ClassName": className,
			"InstanceID": detection.Instance,
			"Area": detection.Area,
			"Bottom": detection.Bottom,
			"CenterX": detection.Center[0],
			"CenterY": detection.Center[1],
			"Confidence": detection.Confidence,
			"Height": detection.Height,
			"Width": detection.Width,
			"Left": detection.Left,
			"Right": detection.Right,
			"Top": detection.Top,
			"Timestamp": time.time(),
			"TargetX": targetX,
			"TargetY": targetY,
			"TargetDistance": targetDistance,
			"AreaPercent": areaPercent
		}
		ntIntakeDetections.append(ntDetection)

		if targetDistance < closestIntakeDetectionDistance:
			closestIntakeDetection = ntDetection
			closestIntakeDetectionDistance = targetDistance

	sd.putString("Intake Detections", json.dumps(ntIntakeDetections))

	if closestIntakeDetection is None:
		sd.putString("Intake Closest Detection", "")
	else:
		sd.putString("Intake Closest Detection", json.dumps(closestIntakeDetection))
		drawCrossHairs(img, closestIntakeDetection["CenterX"], closestIntakeDetection["CenterY"], 255, 255, 255, 255, 120 * captureScale, 30 * captureScale, 1)

	closestGripperDetection = None
	closestGripperDetectionDistance = 10000

	ntGripperDetections = []
	for detection in detections2:

		areaPercent = ((detection.Area / captureArea) * 100)
		targetX = detection.Center[0] - crosshairX
		targetY = detection.Center[1] - crosshairY
		targetDistance = math.sqrt(targetX**2 + targetY**2)

		className = "cubes"
		if (detection.ClassID == 2):
			className = "cones"

			imgCropped = jetson.utils.cudaAllocMapped(width=detection.Width, height=detection.Height, format=img.format)
			crop_roi = (detection.Left * 0.9, detection.Top * 0.9, detection.Right * 0.9, detection.Bottom * 0.9)
			jetson.utils.cudaCrop(img2, imgCropped, crop_roi)
			bgr_img = jetson.utils.cudaAllocMapped(width=imgCropped.width, height=imgCropped.height, format='bgr8')
			jetson.utils.cudaConvertColor(imgCropped, bgr_img)
			jetson.utils.cudaDeviceSynchronize()
			cv_img = jetson.utils.cudaToNumpy(bgr_img)
			hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
			height, width, _ = hsv_img.shape
			upper_half = hsv_img[0:height//2, :]
			lower_half = hsv_img[height//2:, :]

			lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
			upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

			mask_upper = cv2.inRange(upper_half, lower_yellow, upper_yellow)
			mask_lower = cv2.inRange(lower_half, lower_yellow, upper_yellow)

			yellow_pixels_upper = cv2.countNonZero(mask_upper)
			yellow_pixels_lower = cv2.countNonZero(mask_lower)
			shape = None
			if (yellow_pixels_upper > yellow_pixels_lower):
				sd.putString("Cone Rotation", "Upside Down")
			elif (yellow_pixels_lower > yellow_pixels_upper):
				sd.putString("Cone Rotation", "Normal")
			else:
				sd.putString("Cone Rotation", "Unknown")

			del imgCropped
			del bgr_img
			del cv_img

		else:
			sd.putString("Cone Rotation", "Not Cone")

		ntDetection = {
			"ClassID": detection.ClassID,
			"ClassName": className,
			"InstanceID": detection.Instance,
			"Area": detection.Area,
			"Bottom": detection.Bottom,
			"CenterX": detection.Center[0],
			"CenterY": detection.Center[1],
			"Confidence": detection.Confidence,
			"Height": detection.Height,
			"Width": detection.Width,
			"Left": detection.Left,
			"Right": detection.Right,
			"Top": detection.Top,
			"Timestamp": time.time(),
			"TargetX": targetX,
			"TargetY": targetY,
			"TargetDistance": targetDistance,
			"AreaPercent": areaPercent
		}
		ntGripperDetections.append(ntDetection)

		if targetDistance < closestGripperDetectionDistance:
			closestGripperDetection = ntDetection
			closestGripperDetectionDistance = targetDistance

	sd.putString("Gripper Detections", json.dumps(ntGripperDetections))
	sd.putNumber("Net FPS", net.GetNetworkFPS())

	if closestGripperDetection is None:
		sd.putString("Gripper Closest Detection", "")
	else:
		sd.putString("Gripper Closest Detection", json.dumps(closestGripperDetection))
		drawCrossHairs(img2, closestGripperDetection["CenterX"], closestGripperDetection["CenterY"], 255, 255, 255, 255, 120 * captureScale, 30 * captureScale, 1)

	drawCrossHairs(img, crosshairX, crosshairY, 0, 255, 0, 255, 120 * captureScale, 30 * captureScale, 1)
	drawCrossHairs(img2, crosshairX, crosshairY, 0, 255, 0, 255, 120 * captureScale, 30 * captureScale, 1)


	if display is not None:
		display.Render(img)
		display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


	if smallImg is None:
		smallImg = jetson.utils.cudaAllocMapped(width=args.stream_width, height=args.stream_height, format=img.format)
		smallImg2 = jetson.utils.cudaAllocMapped(width=args.stream_width, height=args.stream_height, format=img.format)
	
	jetson.utils.cudaResize(img, smallImg)
	jetson.utils.cudaResize(img2, smallImg2)
	
	del img
	del img2


	if bgrSmallImg is None:
		bgrSmallImg = jetson.utils.cudaAllocMapped(width=args.stream_width, height=args.stream_height, format="bgr8")
		bgrSmallImg2 = jetson.utils.cudaAllocMapped(width=args.stream_width, height=args.stream_height, format="bgr8")

	jetson.utils.cudaConvertColor(smallImg, bgrSmallImg)
	jetson.utils.cudaConvertColor(smallImg2, bgrSmallImg2)

	jetson.utils.cudaDeviceSynchronize()

	numpyImg = jetson.utils.cudaToNumpy(bgrSmallImg, args.stream_width, args.stream_height, 4)
	numpyImg2 = jetson.utils.cudaToNumpy(bgrSmallImg2, args.stream_width, args.stream_height, 4)

	csSource.putFrame(numpyImg)
	csSource2.putFrame(numpyImg2)

	del numpyImg
	del numpyImg2

	cpu = float(os.popen("cat /sys/devices/virtual/thermal/thermal_zone0/temp").read()) / 1000
	gpu = int(os.popen("cat /sys/devices/virtual/thermal/thermal_zone1/temp").read()) / 1000
	endTime = time.time()
	elapseTime = (endTime - startTime) * 1000
	startTime = endTime
	sd.putNumber("Latency", elapseTime)
	sd.putNumber("Pipeline FPS", 1000 / elapseTime)
	sd.putNumber("CPU Temp", cpu)
	sd.putNumber("GPU Temp", gpu)