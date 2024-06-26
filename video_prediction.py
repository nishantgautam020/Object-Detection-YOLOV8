import os
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

def model_predict():
    # Define the new video path
    video_path = os.path.join('/Users/nishantgautam/Desktop/buro/video/', 'designs.mp4')
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Define the path to the custom model
    model_path = '/Users/nishantgautam/Desktop/yolo-results-windows/train2-epoch3/weights/best.pt'

    # Load the custom model
    model = YOLO(model_path)

    threshold = 0.5

    while ret:

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
