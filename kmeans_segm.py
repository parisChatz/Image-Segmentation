import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# TODO do a barchart of pic and dice
# TODO tune cv
# TODO create functions

# Custom libs
from my_paths import directory, save_dir

metrics = {'filename': [], 'dice': [], 'mean_dice': []}
cols = ['filename', 'dice', 'mean_dice']
metrics = pd.DataFrame(data=metrics)
dice_sum = 0
counter = 0
max_mean_dice = 0
dice_mean = 0
kernel_best = (0, 0)
kernel_sizes = [5, 7, 9, 11, 13, 15]


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        print("Different image sizes")
        exit()

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


for curr_kernel_size in kernel_sizes:
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
        # if filename == "ISIC_0000220.jpg":
            print("_" * 20)
            print('Processing Image {}'.format(filename))

            img_path = os.path.join(directory, filename)

            image = cv2.imread(img_path)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # image2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            height, width, channels = image.shape
            kernel_size = (curr_kernel_size, curr_kernel_size)
            kernel = np.ones(kernel_size, np.uint8)

            # Perform closing to remove hair and blur the image
            closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
            # closing2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

            blur = cv2.blur(closing, kernel_size)
            blur = cv2.medianBlur(blur, kernel_size[0])
            blur = cv2.GaussianBlur(blur, kernel_size, 2)

            image2 = blur

            # image2 = cv2.blur(image2, (15, 15))
            # image2 = cv2.GaussianBlur(image2, (15, 15), 0)
            # image2 = cv2.GaussianBlur(image2, (15, 15), 0)
            # image2 = cv2.medianBlur(image2, 15)
            # image2 = cv2.bilateralFilter(image2, 15, 15, 15)

            # # Flatten the 2D image array into an MxN feature vector, where M is
            # # the number of pixels and N is the dimension (number of channels).
            # reshaped = image2.reshape(image2.shape[0] * image2.shape[1], 1)
            pixel_values = image2.reshape((image2.shape[0] * image2.shape[1], 3))

            # convert to float
            pixel_values = np.float32(pixel_values)
            # reshaped = np.float32(reshaped)

            # define stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

            # number of clusters (K)
            k = 2
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 40, cv2.KMEANS_PP_CENTERS)

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

            #################################################

            grid = plt.GridSpec(1, 2)

            plt.figure(figsize=(10, 8))
            plt.subplot(grid[0, 0])  # plot in the first cell
            plt.subplots_adjust(hspace=.5)
            plt.title("hist")
            plt.hist(mask.ravel(), 256, [0, 256])

            plt.subplot(grid[:, 1])  # plot in the 4th cell
            plt.title("RGB_image")
            plt.imshow(mask, cmap='gray')
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            #################################################

            ret, mask = cv2.threshold(mask, 94, 255, cv2.THRESH_BINARY)  # THIS IS THE FINAL MASK!!!!
            # Search for contours and select the biggest one
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key=cv2.contourArea)

            h, w = image.shape[:2]
            mask = np.zeros((h, w), np.uint8)

            # Draw the contour on the new mask and perform the bitwise operation
            mask = cv2.drawContours(mask, [cnt], -1, 255, -1)

            # # Plotting
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # # IMAGE WITH MASK PLOT
            # new_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
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

            gt_path = os.path.join('/home/paris/PycharmProjects/cv_assesment/cv_files/skin lesion dataset/GT')
            gt_filename = str(filename.split('.')[0]) + '_Segmentation.png'
            image_gt = cv2.imread(os.path.join(gt_path, gt_filename), cv2.IMREAD_GRAYSCALE)

            dice_score = dice(mask, image_gt)
            #
            # plt.subplot(grid[1, 1])
            # plt.title(" ground truth {}".format(dice_score))
            # plt.imshow(image_gt, cmap='gray')
            #
            # fig.savefig(save_dir+filename)
            #
            # plt.show(block=False)
            # plt.pause(2)
            # plt.close()

            dice_sum += dice_score
            counter += 1
            dice_mean = dice_sum / counter
            print(dice_score, dice_sum, counter, dice_mean)

            temp_metrics = pd.Series([filename, dice_score, dice_mean],
                                     index=['filename', 'dice', 'mean_dice'])
            metrics = metrics.append(temp_metrics, ignore_index=True)

            print(metrics.head(20))
            print("_" * 20)

    if dice_mean > max_mean_dice:
        max_mean_dice = dice_mean
        kernel_best = kernel_size

        # metrics[cols].to_csv(save_dir + "metrics.csv", index=False)
    print(kernel_best)
