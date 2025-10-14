import numpy as np
import cv2

video_path = r'C:\Users\51388\Videos\h1-stair-v0\video_0.mp4'
with open(video_path, 'rb') as f:
    video_data = f.read()
    mat = cv2.imdecode(np.frombuffer(video_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('Video', mat)
    cv2.waitKey(0)