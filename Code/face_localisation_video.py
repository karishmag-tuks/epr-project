import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.CAP_PROP_BUFFERSIZE
height, width = 480, 640


def convert_to_grayscale(img):
    grayscale_img = np.array(0.2989 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype("uint8")
    return grayscale_img


while True:
    _, frame = cap.read()
    grayscale = convert_to_grayscale(frame)
    cv2.imshow("Frame", grayscale)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
