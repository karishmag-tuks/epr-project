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


def gaussian_kernel(size, sigma=2.5):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def non_maximum_suppression(img, direction):
    M, N = img.shape
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180
    angle [angle < 22.5] += 22.5
    angle = (angle[:, :] - 22.5)// 45
    
    for i in range(1,M-1):
        for j in range(1,N-1):
                mag = img[i,j]
                if ((angle[i,j] == 0 or angle[i,j] == 4) and (img[i, j+1] > mag or img[i, j-1] > mag)
                or (angle[i,j] == 1 and (img[i+1, j-1] > mag or  img[i-1, j+1] > mag))
                or (angle[i,j] == 2 and (img[i+1, j] > mag or img[i-1, j] > mag))
                or (angle[i,j] == 3 and (img[i-1, j-1] > mag or img[i+1, j+1] > mag))):
                    img[i,j] = 0

               

            
    
    return img

# def non_maximum_suppression(gradient, angle):
    M, N = gradient.shape

    angle[angle < 0] += np.pi
    angle  = np.round(angle / (np.pi / 4))
    
    for i in range(1,M-1):
        for j in range(1,N-1):    
            mag = gradient[i, j]
            rangle = angle[i, j]
            if ((rangle == 0 or rangle == 4) and (gradient[i, j+1] < mag or gradient[i, j-1] < mag)
                    or (rangle == 1 and (gradient[i+1, j-1] < mag or gradient[i-1, j+1] < mag))
                    or (rangle == 2 and (gradient[i+1, j] < mag or gradient[i-1, j]< mag))
                    or (rangle == 3 and (gradient[i-1, j-1] < mag or gradient[i+1, j+1] < mag))):
                gradient[i, j] = 0     
    
    return gradient

def double_threshold(img, low_threshold_ratio, high_threshold_ratio):
    high_threshold = img.max() * high_threshold_ratio
    low_threshold  =  low_threshold_ratio * high_threshold

    M, N = img.shape
    res = np.zeros((M,N), dtype="uint8")
    
    weak = np.uint8(50)
    strong = np.uint8(255)
    
    strong_i, strong_j = np.nonzero(img >= high_threshold)    
    weak_i, weak_j = np.nonzero((img <= high_threshold) & (img >= low_threshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res

def hysteresis(img, weak, strong=255):
    weak_i, weak_j = np.nonzero(img == weak)
    for x in range(0, len(weak_i)):
            i, j = weak_i[x], weak_j[x]

            if ((i == 0 or i == img.shape[0]-1) and (j == 0 or i == img.shape[1]-1)):
                continue
            else :
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
              
    return blurred_image

    
gaussian_kernel = gaussian_kernel(3,1)
K_x = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], np.float32)

K_y =  np.array([[1, 2, 1],
                [0, 0, 0],
                [-1,-2,-1]], np.float32)

img = cv2.imread("./data/syntheseyes_data/f05/f05_51_-0.1963_-0.1963.png")
gray_image = convert_to_grayscale(img)
gaussian_blur = convolution(gray_image,gaussian_kernel)
eq_image = histogram_equalisation(gray_image)


sobel_x = convolution(eq_image, K_x)
sobel_y = convolution(eq_image,K_y)

G_mag =  np.sqrt((sobel_x ** 2.0)+(sobel_y ** 2.0))
G_mag = np.array((G_mag/G_mag.max()) * 255, dtype="uint8")
G_dir = np.arctan2(sobel_y, sobel_x)

non_max_img = non_maximum_suppression(G_mag, G_dir)
cv2.imshow("non_max_img",np.array(non_max_img).astype("uint8"))
double_thresh_img = double_threshold(non_max_img, 0.27
                                     , 0.3)
hysteresis_img = hysteresis(double_thresh_img,50,255)

cv2.imshow("hystersis",hysteresis_img)


# # def display_circles(a, img):
#     fig = plt.figure()
#     plt.imshow(img)
#     circle_coordinates = np.argwhere(a)                                          #Extracting the circle information
#     circle = []
#     for r,x,y in circle_coordinates:
#         circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
#         fig.add_subplot(111).add_artist(circle[-1])
#     plt.show()


# def detect_circles(img,threshold,region,radius = None):
    # (M,N) = img.shape
    # if radius == None:
    #     r_max = np.max((M,N))
    #     r_min = 1
    # else:
    #     [r_max,r_min] = radius

    # R = r_max - r_min
    # #Initializing accumulator array.
    # #Accumulator array is a 3 dimensional array with the dimensions representing
    # #the radius, X coordinate and Y coordinate resectively.
    # #Also appending a padding of 2 times R_max to overcome the problems of overflow
    # A = np.zeros((r_max,M+2*r_max,N+2*r_max))
    # B = np.zeros((r_max,M+2*r_max,N+2*r_max))

    # #Precomputing all angles to increase the speed of the algorithm
    # theta = np.arange(0,360)*np.pi/180
    # edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    # for val in range(R):
    #     r = r_min+val
    #     #Creating a Circle Blueprint
    #     bprint = np.zeros((2*(r+1),2*(r+1)))
    #     (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
    #     for angle in theta:
    #         x = int(np.round(r*np.cos(angle)))
    #         y = int(np.round(r*np.sin(angle)))
    #         bprint[m+x,n+y] = 1
    #     constant = np.argwhere(bprint).shape[0]
    #     for x,y in edges:                                                       #For each edge coordinates
    #         #Centering the blueprint circle over the edges
    #         #and updating the accumulator array
    #         X = [x-m+r_max,x+m+r_max]                                           #Computing the extreme X values
    #         Y= [y-n+r_max,y+n+r_max]                                            #Computing the extreme Y values
    #         A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
    #     A[r][A[r]<threshold*constant/r] = 0

    # for r,x,y in np.argwhere(A):
    #     temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
    #     try:
    #         p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
    #     except:
    #         continue
    #     B[r+(p-region),x+(a-region),y+(b-region)] = 255

    # return B[:,r_max:-r_max,r_max:-r_max]

# find_circle = detect_circles(hysteresis_img, 25, 2)
# print(find_circle)

# display_circles(find_circle, hysteresis_img)



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


# plt.figure(1)

# plt.imshow(hysteresis_img)
# plt.show()

# cv2.imshow(eq_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
  