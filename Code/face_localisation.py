import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import warnings
from PIL import Image
warnings.filterwarnings("error")

capture = cv2.VideoCapture(0)
ret, img = capture.read()



cv2.imshow('original ', img)



def convert_to_grayscale():
    grayscale_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            grayscale_img[i][j] = 0.2989 * img[i][j][2] + \
                0.587 * img[i][j][2] + 0.114 * img[i][j][0]
    return grayscale_img

    
image = convert_to_grayscale()


def find_distribution(find_array_dist):
    # sns.distplot(find_array_dist, hist=False)
    return find_array_dist.flatten().std(), find_array_dist.flatten().mean()
    # plt.show()

def find_face_global_thresholding(image_input, std, mean):
    bounded_image = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if mean - std < image_input[i][j] < mean + std:
                bounded_image[i][j] = 0
            else:
                bounded_image[i][j] = 255

    return bounded_image

def find_face_local_mean_thresholding(image, neighbours, constant1):
    bounded_image = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    n = int(neighbours/2)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            store = image[ max(i-n,0):min(i+n,img.shape[0]) +1 ,max(j-n,0):min(j+n,img.shape[1])+1 ].flatten()
            pixel = np.rint(np.sum(store)/store.shape[0])
            bounded_image[i][j] = 255 if (pixel > constant1 ) else 0

    return bounded_image

binarised_image = find_face_local_mean_thresholding(image, 6, 160)
cv2.imshow('image', binarised_image)

def label_connected_components(image):
    labeled_image = np.zeros_like(image)
    current_label = 1

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 0 and labeled_image[y, x] == 0:
                stack = [(y, x)]

                while stack:
                    y, x = stack.pop()
                    labeled_image[y, x] = current_label

                    neighbors = [
                        (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)
                    ]

                    for ny, nx in neighbors:
                        if (
                            0 <= ny < image.shape[0] and
                            0 <= nx < image.shape[1] and
                            image[ny, nx] == 0 and
                            labeled_image[ny, nx] == 0
                        ):
                            stack.append((ny, nx))

                current_label += 1

    return labeled_image

# Label connected components in the binarized image
labeled_image = label_connected_components(binarised_image)

# Extract bounding boxes for each connected component
def extract_bounding_boxes(labeled_image):
    bounding_boxes = {}

    for y in range(labeled_image.shape[0]):
        for x in range(labeled_image.shape[1]):
            label = labeled_image[y, x]
            if label != 0:
                if label not in bounding_boxes:
                    bounding_boxes[label] = {
                        'min_x': x, 'max_x': x,
                        'min_y': y, 'max_y': y
                    }
                else:
                    bounding_boxes[label]['min_x'] = min(bounding_boxes[label]['min_x'], x)
                    bounding_boxes[label]['max_x'] = max(bounding_boxes[label]['max_x'], x)
                    bounding_boxes[label]['min_y'] = min(bounding_boxes[label]['min_y'], y)
                    bounding_boxes[label]['max_y'] = max(bounding_boxes[label]['max_y'], y)
    


    return bounding_boxes

# Extract bounding boxes
bounding_boxes = extract_bounding_boxes(labeled_image)

# Crop and save each element
for label, box in bounding_boxes.items():
    element = img[box['min_y']:box['max_y']+1, box['min_x']:box['max_x']+1]
    element_image = Image.fromarray(element)
    element_image.save(f'element_{label}.png')

# print("Elements cropped and saved.")

# cv.imwrite('mean_adaptive.jpg',mean_adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()
