import cv2
import numpy as np
import os
import pandas as pd
import imutils


# Custom libs
from my_paths import directory, save_dir
from dice import dice
from plots import plot_results_k

metrics = {'filename': [], 'dice': [], 'mean_dice': []}
cols = ['filename', 'dice', 'mean_dice']
metrics = pd.DataFrame(data=metrics)
max_mean_dice = 0
kernel_best = (0, 0)
kernel_sizes = [13]
dice_array = []


for curr_kernel_size in kernel_sizes:
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print("_" * 20)
            print('Processing Image {}'.format(filename))

            img_path = os.path.join(directory, filename)

            image2 = cv2.imread(img_path)
            image = cv2.imread(img_path)

            height, width, channels = image2.shape

            # Find image center
            img_cy, img_cx = height/2, width/2
            kernel_size = (curr_kernel_size, curr_kernel_size)
            kernel = np.ones(kernel_size, np.uint8)

            # Perform closing to remove hair and blur the image
            closing = cv2.morphologyEx(image2, cv2.MORPH_CLOSE, kernel, iterations=2)
            closing2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

            blur = cv2.medianBlur(closing2, kernel_size[0])
            blur = cv2.GaussianBlur(blur, kernel_size, 2)

            image2 = blur
            # image2 = cv2.bilateralFilter(image2, 15, 15, 15)

            # Flatten the 2D image array into an MxN feature vector, where M is
            # the number of pixels and N is the dimension (number of channels).
            pixel_values = image2.reshape((image2.shape[0] * image2.shape[1], 3))

            # convert to float
            pixel_values = np.float32(pixel_values)

            # define stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

            # number of clusters (K)
            k = 2
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 40, cv2.KMEANS_RANDOM_CENTERS)

            # convert back to 8 bit values
            centers = np.uint8(centers)

            # flatten the labels array
            labels = labels.flatten()

            # convert all pixels to the color of the centroids
            segmented_image = centers[labels.flatten()]

            # reshape back to the original image dimension
            segmented_image = segmented_image.reshape(image2.shape)
            mask1 = cv2.bitwise_not(segmented_image)
            mask = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(mask, max(np.unique(mask)) - 10, 255,
                                      cv2.THRESH_BINARY)  # THIS IS THE FINAL MASK!!!!

            # Search for contours and select the biggest one
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(contours)

            # Find the center of the contours and 'keep' the one closer to the center of the image
            for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                d = np.sqrt(np.power(cX - img_cx, 2) + np.power(cY - img_cy, 2))
                # print(cX, cY, img_cx, img_cy, d)
                if d <= 300:
                    cnt = c

            h, w = image2.shape[:2]
            mask = np.zeros((h, w), np.uint8)

            # Draw the contour on the new mask and perform the bitwise operation
            mask = cv2.drawContours(mask, [cnt], -1, 255, -1)

            gt_path = os.path.join('cv_files/skin lesion dataset/GT')
            gt_filename = str(filename.split('.')[0]) + '_Segmentation.png'
            image_gt = cv2.imread(os.path.join(gt_path, gt_filename), cv2.IMREAD_GRAYSCALE)

            dice_score = dice(mask, image_gt)
            dice_array.append(dice_score)
            dice_mean = np.mean(dice_array)

            # Plotting
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # IMAGE WITH MASK PLOT
            plot_results_k(image_rgb, image_gt, mask, dice_score, filename, True)

            # Save metrics
            temp_metrics = pd.Series([filename, dice_score, dice_mean],
                                     index=['filename', 'dice', 'mean_dice'])
            metrics = metrics.append(temp_metrics, ignore_index=True)

            print(metrics.head(60))
            print("_" * 20)

    if dice_mean > max_mean_dice:
        max_mean_dice = dice_mean
        kernel_best = kernel_size

    print(kernel_best)

# Save metrics as CSV
# metrics[cols].to_csv(save_dir + "metrics.csv", index=False)
