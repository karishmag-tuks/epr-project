import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_to_grayscale(img):
    grayscale_img = np.array(0.2989 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype("uint8")
    return grayscale_img

def histogram_equalisation(img):
    histogram_array = np.bincount(img.flatten(), minlength=256).tolist()
    sum_array = np.sum(histogram_array)
    histogram_array = histogram_array / sum_array
    chistogram_array = np.array([np.sum(histogram_array[0:i+1])  for i in range(len(histogram_array))])
    transform_map = np.floor(chistogram_array * 255).astype("uint8")
    return np.array( [[ transform_map[img[i,j]] for i in range(img.shape[0])] for j in range(img.shape[1])])


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def non_maximum_suppression(img, direction):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            
    
    return Z



def double_threshold(img, low_threshold_ratio, high_threshold_ratio):
    high_threshold = img.max() * high_threshold_ratio
    low_threshold  =  low_threshold_ratio * high_threshold

    M, N = img.shape
    res = np.zeros((M,N), dtype="uint8")
    
    weak = np.int32(50)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= high_threshold)    
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
               
    return img

def convolution(image, kernel):
    offset = int(kernel.shape[0]/2)
    blurred_image = np.zeros((image.shape[0], image.shape[1]), np.float32)
    for i in range(0+offset, image.shape[0]-offset):
         for j in range(0+offset, image.shape[1]-offset):                    
             image_block = image[i-offset:i+offset +1, j-offset:j+offset+1]
             blurred_image[i,j] = sum((image_block * kernel).flatten()) 
              
    return blurred_image;

gaussian_kernel = gaussian_kernel(5,1.4)
K_x = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], np.float32)

K_y =  np.array([[1, 2, 1],
                [0, 0, 0],
                [-1,-2,-1]], np.float32)

img = cv2.imread("./data/syntheseyes_data/f05/f05_42_-0.0000_-0.1963.png")
gray_image = convert_to_grayscale(img)

eq_image = histogram_equalisation(gray_image)

gaussian_blur = convolution(eq_image,gaussian_kernel)
sobel_x = convolution(gaussian_blur, K_x)
sobel_y = convolution(gaussian_blur,K_y)

G_mag =  np.sqrt((sobel_x ** 2.0)+(sobel_y ** 2.0))
G_mag = np.array((G_mag/G_mag.max()) * 255, dtype="uint8")
G_dir = np.arctan2(sobel_y, sobel_x)

non_max_img = non_maximum_suppression(G_mag, G_dir)

double_thresh_img = double_threshold(non_max_img, 0.02, 0.055)
hysteresis_img = hysteresis(double_thresh_img,50,255)

# cv2.imshow("double_thresh", hysteresis_img)

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()

   
# gray_image = convert_to_grayscale(img)
# gaussian_blur = convolution(gray_image,kernel)
# cv2.imshow("image", img)
# cv2.imshow("gray image", gray_image)
# cv2.imshow("blurred image", gaussian_blur)
# cv2.imshow("sobel_x", sobel_x)
# cv2.imshow("sobel_y", sobel_y)


plt.figure(1)

plt.imshow(hysteresis_img)
plt.show()

# cv2.imshow(eq_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
  