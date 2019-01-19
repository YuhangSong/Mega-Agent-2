import cv2
video_writer = cv2.VideoWriter(
    '../results/test_cv.avi',
    cv2.VideoWriter_fourcc('M','J','P','G'),
    5,
    (100,100),
    False
)
import numpy as np
state_img = np.random.randint(0, high=254, size=(100,100)).astype(np.uint8)

for i in range(100):
    video_writer.write(state_img)
video_writer.release()
