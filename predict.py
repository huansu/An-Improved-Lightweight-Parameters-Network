import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    capture = cv2.VideoCapture(0)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Connection failed. Please try again!")

    fps = 0.0
    while(True):
        t1 = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        # BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # convert to Image
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(yolo.detect_image(frame))
        # RGBtoBGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff

        if c==27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
