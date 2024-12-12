import cv2
import numpy as np


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    optical_flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return optical_flow

def draw_optical_flow(optical_flow, image, step=5):
    """
    Draw the optical flow on the image
    :param optical_flow:
    :param image:
    :param step: step size for the lines
    :return:
    """
    h, w = image.shape[:2]  #Height and width of the image
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = optical_flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    if len(image.shape) == 2:  # Check if the image is already grayscale
        optical_flow_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        optical_flow_image = image.copy()

    for (x1, y1), (x2, y2) in lines:
        cv2.line(optical_flow_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(optical_flow_image, (x1, y1), 1, (0, 255, 0), -1)
    return optical_flow_image
