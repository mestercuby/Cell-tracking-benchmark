import os
import random

import cv2
from ultralytics import YOLO
import argparse
from tracker import Tracker

def main(args):
    video_path = args.video
    video_out_path = args.video_out

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                              (frame.shape[1], frame.shape[0]))

    model = YOLO("pretrained/yolo11n.pt")

    tracker = Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    detection_threshold = 0.5
    while ret:

        results = model(frame)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap_out.write(frame)
        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "demo.mp4", help='video file name')
parser.add_argument('--video_out', type=str, default = "demo/road_out.mp4", help='video out file name')


args = parser.parse_args()
main(args)
