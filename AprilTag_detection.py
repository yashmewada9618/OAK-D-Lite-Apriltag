#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from pupil_apriltags import Detector
# Create pipeline
pipeline = dai.Pipeline()

# Define source and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutPreview = pipeline.create(dai.node.XLinkOut)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')
xoutPreview.setStreamName("preview")

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camRgb.setPreviewSize(960, 540)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(True)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Linking
camRgb.preview.link(xoutPreview.input)
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)
# Connect to device and start pipeline


def draw_tags(
        image,
        tags):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # 中心
        cv2.circle(image,
                   (center[0], center[1]), 5, (255, 0, 255), 2)

        # 各辺
        cv2.line(image, (corner_01[0], corner_01[1]),
                 (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_02[0], corner_02[1]),
                 (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_03[0], corner_03[1]),
                 (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image, (corner_04[0], corner_04[1]),
                 (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        cv2.putText(image, str(tag_id), (center[0] - 50, center[1] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    return image


at_detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)
with dai.Device(pipeline) as device:
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    preview = device.getOutputQueue('preview')
    while True:
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()
        previewFrame = preview.get()
        # if inLeft is not None:
        #     mono_frame = inLeft.getCvFrame()
        # if inRight is not None:
        #     mono_frame = inRight.getCvFrame()
        if inLeft is not None:
            tags = at_detector.detect(
                inLeft.getCvFrame(),
                estimate_tag_pose=True,
                camera_params=([0.1, 0.1, 0.1, 0.1]),
                tag_size=0.05,
            )
            debug_image = draw_tags(previewFrame.getCvFrame(), tags)
            cv2.imshow("left", debug_image)

        if inRight is not None:
            tags = at_detector.detect(
                inRight.getCvFrame(),
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            debug_image = draw_tags(previewFrame.getCvFrame(), tags)
            cv2.imshow("right", debug_image)
        # Get BGR frame from NV12 encoded video frame to show with opencv
        #cv2.imshow("video", videoFrame.getCvFrame())
        # Show 'preview' frame as is (already in correct format, no copy is made)
        # cv2.imshow("preview", previewFrame.getCvFrame())
        # cv2.imshow("right", inRight.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            break
