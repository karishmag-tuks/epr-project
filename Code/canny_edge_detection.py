import numpy as np
import cv2



def convert_to_grayscale(img):
    grayscale_img = np.array(0.2989 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype("uint8")
    return grayscale_img

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

kernel = gaussian_kernel(3,1)


def convolution(image, kernel):
    offset = int(kernel.shape[0]/2)
    blurred_image = np.zeros((image.shape[0], image.shape[1])).astype("uint8")
    for i in range(0+offset, image.shape[0]-offset):
         for j in range(0+offset, image.shape[1]-offset):                    
             image_block = image[i-offset:i+offset +1, j-offset:j+offset+1]
             blurred_image[i,j] = sum(image_block.flatten() * kernel.flatten()) 
              
    return blurred_image;


img = cv2.imread("./data/syntheseyes_data/f01/f01_42_-0.0000_-0.1963.png")
gray_image = convert_to_grayscale(img)
gaussian_blur = convolution(gray_image,kernel)

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()


while (True):    
    # gray_image = convert_to_grayscale(img)
    # gaussian_blur = convolution(gray_image,kernel)
    # cv2.imshow("image", img)
    # cv2.imshow("gray image", gray_image)
    cv2.imshow("blurred image", gaussian_blur)
    if cv2.waitKey(1)  == 27:
        break