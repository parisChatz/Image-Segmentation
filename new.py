from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import numpy as np
import argparse

# Custom libs
from my_paths import directory

rescale = False

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory, filename)
        image = cv2.imread(img_path)

        # if rescale:
        #     scale_percent = 60  # percent of original size
        #     width = int(image.shape[1] * scale_percent / 100)
        #     height = int(image.shape[0] * scale_percent / 100)
        #     dim = (width, height)
        #     image = cv2.resize(image, dim)  # , interpolation=cv2.INTER_AREA)

        plt.imshow(image)
        # plt.show()

        print('image: ', image.shape)
        image_x = image.shape[0]
        image_y = image.shape[1]

        # Remember images in BGR format
        image_channel_r = image[:, :, 2]
        image_channel_g = image[:, :, 1]
        image_channel_b = image[:, :, 0]

        # channels = [image_channel_r, image_channel_g, image_channel_b]
        # for i in channels:
        #     plt.imshow(i, cmap='gray')
        #     plt.show(block=False)
        #     plt.pause(1)
        #     plt.close()

        ret_r, thresh_r = cv2.threshold(image_channel_r, 150, 255, cv2.THRESH_BINARY_INV)
        ret_g, thresh_g = cv2.threshold(image_channel_g, 125, 255, cv2.THRESH_BINARY_INV)
        ret_b, thresh_b = cv2.threshold(image_channel_b, 120, 255, cv2.THRESH_BINARY_INV)

        # plt.figure(1)
        # plt.imshow(thresh_r, 'gray')
        # plt.figure(2)
        # plt.imshow(thresh_g, 'gray')
        # plt.figure(3)
        # plt.imshow(thresh_b, 'gray')
        # plt.show()

        # add masks
        result = cv2.add(thresh_r, thresh_b, thresh_g)
        gray = rgb2gray(result)
        plt.imshow(gray, cmap='gray')
        # plt.show()

        mask_inv = cv2.bitwise_not(result)
        new_img = cv2.bitwise_and(image, image, mask=result)

        plt.imshow(new_img)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        # todo try color filtering cause many images are cold or hot
        # todo remove small blemishes
        # todo edge detection to get only the canser
