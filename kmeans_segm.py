import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom libs
from my_paths import directory

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory, filename)
        image = cv2.imread(img_path)

        image2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Flatten the 2D image array into an MxN feature vector, where M is
        # the number of pixels and N is the dimension (number of channels).
        reshaped = image2.reshape(image2.shape[0] * image2.shape[1], 1)
        pixel_values = image.reshape((image.shape[0] * image.shape[1], 3))

        # convert to float
        pixel_values = np.float32(pixel_values)
        reshaped = np.float32(reshaped)

        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

        # number of clusters (K)
        k = 2
        _, labels, (centers) = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]
        print("_"*20)
        print('Image {} with labels : {} with shape {}'.format(filename,labels, labels.shape))
        print("_"*20)

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image2.shape)
        mask = cv2.bitwise_not(segmented_image)

        masked_image = np.copy(image2)

        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((image2.shape[0] * image2.shape[1], 1))

        # color (i.e cluster) to disable
        cluster = 0
        masked_image[labels == cluster] = [0]

        # convert back to original shape
        masked_image = masked_image.reshape(image2.shape)
        print("_"*20)
        print(masked_image)
        print("_"*20)
        # show the mask
        # plt.imshow(masked_image, cmap='gray')

        # show the image and mask
        # plt.figure(2)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_img = cv2.bitwise_and(im_rgb, im_rgb, mask=masked_image)

        plt.figure()
        plt.imshow(new_img, cmap='gray')
        plt.figure()
        plt.imshow(im_rgb)
        plt.title(filename)
        plt.show()
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
