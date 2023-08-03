import cv2
import os

capture = cv2.VideoCapture(0)

# name of the display window in openCV
cv2.namedWindow('Face Feed')
try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

currentframe = 0
while currentframe < 10:
    # reading from frame
    ret, frame = capture.read()

    if ret:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
capture.release()
cv2.destroyAllWindows()
