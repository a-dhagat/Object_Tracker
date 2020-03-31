import numpy as np
import cv2

# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

def pre_process(img):
    log_img = np.log(img+1)
    normalize_img = (log_img-np.mean(log_img.flatten()))/(np.std(log_img.flatten() + 1e-5))
    cosine_window = cv2.createHanningWindow(img.shape[:2],cv2.CV_32F)
    output = normalize_img * cosine_window.T
    return output