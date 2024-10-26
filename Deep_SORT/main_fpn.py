import os
import random
import sys
import cv2
from ultralytics import YOLO
import argparse
from tracker import Tracker

sys.path.append('/home/master/Desktop/tensorpack/examples/FasterRCNN')
sys.path.append('/home/master/Desktop/tensorpack/examples/FasterRCNN/dataset')
sys.path.append('/home/master/Desktop/tensorpack/examples/FasterRCNN/utils')
sys.path.append('/home/master/Desktop/tensorpack/examples/FasterRCNN/convert_d2')
sys.path.append('/home/master/Desktop/tensorpack/examples/FasterRCNN/modeling')
import predict

def main(args):
    video_path = args.video
    video_out_path = args.video_out

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                              (frame.shape[1], frame.shape[0]))

    predict.register_coco(predict.cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    predict.register_balloon(predict.cfg.DATA.BASEDIR)

    MODEL = predict.ResNetFPNModel() if predict.cfg.MODE_FPN else predict.ResNetC4Model()
    predict.finalize_configs(is_training=False)
    predict.cfg.TEST.RESULT_SCORE_THRESH = predict.cfg.TEST.RESULT_SCORE_THRESH_VIS
    predcfg = predict.PredictConfig(
        model=MODEL,
        session_init=predict.SmartInit(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])
    model = predict.OfflinePredictor(predcfg)
    print("Model loaded")
    

    tracker = Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    detection_threshold = 0.5
    while ret:

        results=predict.predict_image(frame,model)
        detections = []
        for result in results:
            conf = result.score
            bbox = result.box

            if conf > detection_threshold:
                detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])

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
parser.add_argument('--load', help='load a model for evaluation.', required=True)
args = parser.parse_args()
main(args)