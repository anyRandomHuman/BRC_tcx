import cv2
import os

import numpy as np

env = 'cheetah-run/1'
index = 0


# Path to the video file
video_path = f"videos/{env}/task_{index}.npy"  # Replace with the actual path to your video

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit(1)

# Open the video file
# video = cv2.VideoCapture(video_path)
video = np.load(video_path)
# if not video.isOpened():
#     print("Error: Could not open video.")
#     exit(1)

# Display the video frame by frame
frames= []

for frame in video:
    # Display the frame
    cv2.imshow("Video", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Wait for 30ms and check if the user presses the 'q' key to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
# video.release()
cv2.destroyAllWindows()