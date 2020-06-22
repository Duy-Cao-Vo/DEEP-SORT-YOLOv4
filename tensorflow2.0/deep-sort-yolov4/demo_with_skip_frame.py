#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from collections import deque
from math import sqrt, acos, tan

warnings.filterwarnings('ignore')


def not_count_staff(frame, startX, startY, endX, endY):
    # Define color of shirt following B G R
    boundaries = [
                    ([50, 15, 190], [71, 27, 204])
                ]
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ROI = frame[int(startY+(endY-startY)/4):int(endY-(endY-startY)/4), startX:endX]
    check_null = np.shape(ROI)
    if any(x == 0 for x in check_null):
        return False

    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower)
        upper = np.array(upper)
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(ROI, lower, upper)
        output = cv2.bitwise_and(ROI, ROI, mask=mask)
        # show the images
        if len(np.where(output != 0)[0]) > 100:
            # cv2.imshow("ÁO ĐỎ", output)
            # cv2.waitKey(0)
            return True
    return False


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 2.0
    nn_budget = None
    nms_max_overlap = 3.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True  # show object box blue when detect
    writeVideo_flag = True  # record video ouput

    defaultSkipFrames = 10  # skipped frames between detections

    # set up collection of door
    H1 = 245
    W1 = 370
    H2 = 280
    W2 = 480
    H = None
    W = None

    R = 80 # min R is 56

    def solve_quadratic_equation(a, b, c):
        """ax2 + bx + c = 0"""
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            print ("Phương trình vô nghiệm!")
        elif delta == 0:
            return -b / (2 * a)
        else:
            print ("Phương trình có 2 nghiệm phân biệt!")
            if float ((-b - sqrt (delta)) / (2 * a)) > float ((-b + sqrt (delta)) / (2 * a)):
                return float ((-b - sqrt (delta)) / (2 * a))
            else:
                return float ((-b + sqrt (delta)) / (2 * a))

    def setup_door(H1, W1, H2, W2, R):
        # bước 1 tìm trung điểm của W1, H1 W2, H2
        I1 = (W1 + W2) / 2
        I2 = (H1 + H2) / 2

        # tìm vecto AB
        u1 = W2 - W1
        u2 = H2 - H1

        # AB chính là vecto pháp tuyến của d
        # ta có phương trình trung tuyến của AB
        # y = -(u1 / u2)* x - c/u2
        c = -u1 * I1 - u2 * I2  # tìm c

        # bước 2 tìm tâm O của đường tròn
        al = c / u2 + I2
        # tính D: khoảng cách I và O
        fi = acos(sqrt ((I1 - W1) ** 2 + (I2 - H1) ** 2) / R)
        D = sqrt ((I1 - W1) ** 2 + (I2 - H1) ** 2) * tan(fi)

        O1 = solve_quadratic_equation ((1 + u1 ** 2 / u2 ** 2), 2 * (-I1 + u1 / u2 * al), al ** 2 - D ** 2 + I1 ** 2)
        O2 = -u1 / u2 * O1 - c / u2
        # phương trình 2 nghiệm chỉ chọn nghiệm phía trên

        # Bước 3 tìm các điểm trên đường tròn
        door_dict = dict ()
        for w in range (W1, W2):
            h = O2 + sqrt (R ** 2 - (w - O1) ** 2)
            door_dict[w] = round(h)
        return door_dict

    door_dict = setup_door(H1, W1, H2, W2, R)

    totalFrames = 0
    totalIn = 0

    # create a empty list of centroid to count traffic
    pts = [deque(maxlen=30) for _ in range(9999)]

    file_path = 'D:\\video/[Sala Outside][2020-05-28T16-01-39][2020-05-28T18-02-09].mp4'
    video_capture = cv2.VideoCapture(file_path)

    fps_imutils = imutils.video.FPS().start()

    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output_yolov4.mp4', fourcc, 3, (736, 480))

    while True:
        oke, frame = video_capture.read()  # frame shape 640*480*3
        if not oke:
            break
        
        frame = cv2.resize(frame, (736, 480))
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # calculate video time
        videotime = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Draw a door line
        for w in range(W1, W2):
            cv2.circle(frame, (w, door_dict[w]), 1, (0, 255, 255), -1)
        cv2.circle(frame, (W1, H1), 4, (0, 0, 255), -1)
        cv2.circle(frame, (W2, H2), 4, (0, 0, 255), -1)

        if totalFrames % defaultSkipFrames == 0:
            t2 = time.time()
            boxes, confidence, classes = yolo.detect_image(image)  # average time: 1.2s
            print(time.time() - t2)

            features = encoder(frame, boxes)
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.cls for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for det in detections:
                bbox = det.to_tlbr()
                if show_detections and len(classes) > 0:
                    det_cls = det.cls
                    score = "%.2f" % (det.confidence * 100) + "%"
                    cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3]) - 10), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

            for track in tracker.tracks:
                if not track.is_confirmed():
                    continue
                bbox = track.to_tlbr()

                if not_count_staff(frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])):
                    # adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)
                    cv2.putText(frame, "STAFF", (int(bbox[0]), int(bbox[1]) - 10), 0,
                                1e-3 * frame.shape[0], (0, 0, 255), 1)
                    continue
                else:
                    # adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)

                x = [c[0] for c in pts[track.track_id]]
                y = [c[1] for c in pts[track.track_id]]

                centroid_x = int(((bbox[0]) + (bbox[2])) / 2)
                centroid_y = int(((bbox[1]) + (bbox[3])) / 2)

                if not track.Counted and centroid_x in range(W1, W2):
                    if centroid_y < np.mean(y) and door_dict[centroid_x] > centroid_y and np.max(x) - np.min(x) > 20:
                        totalIn += 1
                        track.Counted = True
                        print(track.track_id, track.Counted)

                cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
                pts[track.track_id].append((centroid_x, centroid_y))

            info = [
                ("Time", "{:.4f}".format (videotime)),
                ("In", totalIn)
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (W - 150, ((i * 20) + 20)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if writeVideo_flag:
                # save a frame
                out.write(frame)

            if show_detections:
                cv2.imshow('People counter', frame)
                # Press Q to stop!
                if cv2.waitKey(1) & 0xFF == ord ('q'):
                    break
        else:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

        fps_imutils.update()

        totalFrames += 1

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if writeVideo_flag:
        out.release()

    video_capture.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
