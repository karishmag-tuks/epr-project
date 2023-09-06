


def convert_to_grayscale():
    grayscale_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            grayscale_img[i][j] = 0.2989 * img[i][j][2] + \
                0.587 * img[i][j][2] + 0.114 * img[i][j][0]
    return grayscale_img

    
image = convert_to_grayscale()