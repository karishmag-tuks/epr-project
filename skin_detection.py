import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# img = cv2.imread("small-joseph.png", cv2.IMREAD_COLOR)
# img2 = cv2.imread("small-joseph.png", cv2.IMREAD_GRAYSCALE)
capture = cv2.VideoCapture(0)
ret, img = capture.read()


def convert_to_grayscale():
    grayscale_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            grayscale_img[i][j] = 0.2989 * img[i][j][2] + 0.587 * img[i][j][2] + 0.114 * img[i][j][0]
    return grayscale_img


def find_distribution(find_array_dist):
    # sns.distplot(find_array_dist, hist=False)

    return find_array_dist.flatten().std(), find_array_dist.flatten().mean()

    # plt.show()


def find_face(find_image_boundary, std, mean):
    bounded_image = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if mean - std < find_image_boundary[i][j] < mean + std:
                bounded_image[i][j] = 0
            else:
                bounded_image[i][j] = 255

    return bounded_image


image = convert_to_grayscale()
cv2.imshow('image', image)
std, mean = find_distribution(image)
print(mean)
cv2.imshow('image2', find_face(image, std, mean))
cv2.waitKey(0)
cv2.destroyAllWindows()
