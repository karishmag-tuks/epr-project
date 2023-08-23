import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os


# img = cv2.imread('data/small-joseph.png', cv2.IMREAD_COLOR)
# img2 = cv2.imread("small-joseph.png", cv2.IMREAD_GRAYSCALE)
capture = cv2.VideoCapture(0)
ret, img = capture.read()


def convert_to_grayscale():
    grayscale_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            grayscale_img[i][j] = 0.2989 * img[i][j][2] + \
                0.587 * img[i][j][2] + 0.114 * img[i][j][0]
    return grayscale_img


def find_distribution(find_array_dist):
    # sns.distplot(find_array_dist, hist=False)
    return find_array_dist.flatten().std(), find_array_dist.flatten().mean()
    # plt.show()


def find_face_global_thresholding(image_input, std, mean):
    bounded_image = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if mean - std < image_input[i][j] < mean + std:
                bounded_image[i][j] = 255
            else:
                bounded_image[i][j] = 0

    return bounded_image

def find_face_local_mean_thresholding(image, neighbours, constant):
    bounded_image = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    n = int(neighbours/2)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            index1 = max(i-n,0)
            index2 = min(i+n,img.shape[0])
            index3 = max(j-n,0)
            index4 = min(j+n,img.shape[1])
            pixel = np.average(image[index1:index2 ][index3:index4 ].flatten())
            bounded_image[i][j] = 255 if (pixel > constant) else 0

    return bounded_image



image = convert_to_grayscale()
# print(image.shape)
cv2.imshow('image', image)
std, mean = find_distribution(image)

cv2.imshow('mean_adaptive',mean_adaptive)
# cv.imwrite('mean_adaptive.jpg',mean_adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()
