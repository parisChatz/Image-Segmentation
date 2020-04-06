import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import pandas as pd
from my_paths import directory, save_dir, gt_path
from dice import dice

dice_sum = 0
counter = 0
dice_mean = 0

metrics = {'filename': [], 'dice': [], 'mean_dice': []}
cols = ['filename', 'dice', 'mean_dice']
metrics = pd.DataFrame(data=metrics)

# Read the image and perfrom an OTSU threshold
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print("_" * 20)
        print('Processing Image {}'.format(filename))

        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kernel_size = (9, 9)
        kernel = np.ones(kernel_size, np.uint8)

        # Perform closing to remove hair and blur the image
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        # closing2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

        blur = cv2.blur(closing, kernel_size)
        blur = cv2.medianBlur(blur, kernel_size[0])
        blur = cv2.GaussianBlur(blur, kernel_size, 2)

        image2 = blur

        # Binarize the image
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #################################################

        grid = plt.GridSpec(1, 2)

        plt.figure(figsize=(10, 8))
        plt.subplot(grid[0, 0])  # plot in the first cell
        plt.subplots_adjust(hspace=.5)
        plt.title("hist")
        plt.hist(thresh.ravel(), 256, [0, 256])

        plt.subplot(grid[:, 1])  # plot in the 4th cell
        plt.title("RGB_image")
        plt.imshow(thresh, cmap='gray')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        #################################################

        ret, mask = cv2.threshold(thresh, 94, 255, cv2.THRESH_BINARY)  # THIS IS THE FINAL MASK!!!!
        # Search for contours and select the biggest one
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        # Draw the contour on the new mask and perform the bitwise operation
        mask = cv2.drawContours(mask, [cnt], -1, 255, -1)

        # # Plotting
        # # IMAGE WITH MASK PLOT
        # new_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        #
        # grid = plt.GridSpec(ncols=2, nrows=2)
        # fig = plt.figure(figsize=(12, 8))
        # plt.subplot(grid[0, 0])
        # plt.title(filename + " RGB image")
        # plt.imshow(image_rgb)
        # plt.subplot(grid[1, 0])
        # plt.title(filename + " mask image")
        # plt.imshow(mask, cmap='gray')
        # plt.subplot(grid[0, 1])
        # plt.title(filename + " segmented image")
        # plt.imshow(new_img, cmap='gray')

        path = os.path.join(gt_path)
        gt_filename = str(filename.split('.')[0]) + '_Segmentation.png'
        image_gt = cv2.imread(os.path.join(path, gt_filename), cv2.IMREAD_GRAYSCALE)

        dice_score = dice(mask, image_gt)

        # plt.subplot(grid[1, 1])
        # plt.title(" ground truth {}".format(dice_score))
        # plt.imshow(image_gt, cmap='gray')

        # fig.savefig(save_dir+filename)

        # plt.show(block=False)
        # plt.pause(2)
        # plt.close()

        dice_sum += dice_score
        counter += 1
        dice_mean = dice_sum / counter
        temp_metrics = pd.Series([filename, dice_score, dice_mean],
                                 index=['filename', 'dice', 'mean_dice'])
        metrics = metrics.append(temp_metrics, ignore_index=True)

        print(metrics.head(60))
        print("_" * 20)

