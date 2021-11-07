# please goto below link to download yolov4.cfg and yolov4.weights,
# wget https://github.com/AlexeyAB/darknet/tree/master/cfg/yolov4.cfg
# wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# and put those files into yolo-coco_v4
# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os


class yolo_object_detection():
# private
    __labelsPath = "./yolo-coco_v4/coco.names"
    __weightsPath = "yolo-coco_v4/yolov4.weights"
    __configPath = "yolo-coco_v4/yolov4.cfg"
    __confidence_setting = 0.5
    __threshold = 0.2

# public
    def __init__(self, target_label):
        np.random.seed(42)
        self.__LABELS = open(self.__labelsPath).read().strip().split("\n")
        self.__COLORS = np.random.randint(0, 255, size=(len(self.__LABELS), 3), dtype="uint8")
        self.__target_label = target_label



    def run_detection(self, frame, ):
        # determine only the *output* layer names that we need from YOLO
        self.__net = cv2.dnn.readNetFromDarknet(self.__configPath, self.__weightsPath)
        # determine only the *output* layer names that we need from YOLO
        ln = self.__net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

        # initialize the width and height of the frames in the video file
        W = None
        H = None

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        crop_width = 1920
        crop_high = 1080
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        # crop small area to improve the effect of yolo
        # crop image size 960*540
        # overlap 40%
        for crop_x in range(0, 2840, 960):
            for crop_y in range(0, 1620, 540):
                # print("crop_y:%d" % crop_y)
                # print("crop_y + crop_high:%d" % (crop_y + crop_high))
                # print("crop_x:%d" % crop_x)
                # print("crop_x + crop_width:%d" % (crop_x + crop_width))

                crop_img = frame[crop_y:crop_y + crop_high, crop_x:crop_x + crop_width]
                blob = cv2.dnn.blobFromImage(crop_img, 1 / 255.0, (416, 416),
                                             swapRB=True, crop=False)
                self.__net.setInput(blob)
                layerOutputs = self.__net.forward(ln)
                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > self.__confidence_setting:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([crop_width, crop_high, crop_width, crop_high])
                            (centerX, centerY, width, height) = box.astype("int")
                            if width < crop_width * 2 / 3:
                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([crop_x + x, crop_y + y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.__net.setInput(blob)
        layerOutputs = self.__net.forward(ln)
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.__confidence_setting:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.__confidence_setting, self.__threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #x = x + int(w/5)
                # draw a bounding box rectangle and label on the frame
                if self.__LABELS[classIDs[i]] == self.__target_label:
                    color = [int(c) for c in self.__COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.__LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite("frame.jpg",frame)
        return boxes


def IOU_check_for_first_frame(src_bbox, adjust_bboxes):
    print("IoU method(first_frame)")
    iou_temp = []
    boxSRCArea = (src_bbox[2] + 1) * (src_bbox[3] + 1)
    for i, adjust_bbox in enumerate(adjust_bboxes):
        xA = max(src_bbox[0], adjust_bbox[0])
        # print("xA:%.2f" % xA)
        yA = max(src_bbox[1], adjust_bbox[1])
        # print("yA:%.2f" % yA)
        xB = min(src_bbox[0] + src_bbox[2], adjust_bbox[0] + adjust_bbox[2])
        # print("xB:%.2f" % xB)
        yB = min(src_bbox[1] + src_bbox[3], adjust_bbox[1] + adjust_bbox[3])
        # print("yB:%.2f" % yB)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # print("interArea:%.2f" % interArea)

        boxADJArea = (adjust_bbox[2] + 1) * (adjust_bbox[3] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction  ground-truth
        # areas - the intersection area
        iou = interArea / float(boxSRCArea + boxADJArea - interArea)
        iou_temp.append(iou)

    # if max(iou_temp) > 0:
    iou_array = np.array(iou_temp)
    index = np.argmax(iou_array)
    print("iou_array:")
    print(iou_array)

    if iou_array[index] > 0.5:
        return adjust_bboxes[index]
    else:
        return src_bbox

