import numpy as np
import cv2
import os
import pandas as pd

from my_paths import directory, save_dir, gt_path
from dice import dice
from plots import plot_results_k

metrics = {'filename': [], 'dice': [], 'mean_dice': []}
cols = ['filename', 'dice', 'mean_dice']
metrics = pd.DataFrame(data=metrics)
max_mean_dice = 0
kernel_best = (0, 0)
kernel_sizes = [13]

dice_array = []

# Read the image and perfrom an OTSU threshold
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print("_" * 20)
        print('Processing Image {}'.format(filename))

        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kernel_size = (13, 13)
        kernel = np.ones(kernel_size, np.uint8)

        # Perform closing to remove hair and blur the image
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        closing2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

        blur = cv2.medianBlur(closing2, kernel_size[0])
        blur = cv2.GaussianBlur(blur, kernel_size, 2)

        image2 = blur

        # Binarize the image
        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ret, mask = cv2.threshold(thresh, max(np.unique(thresh)) - 30, 255,
                                  cv2.THRESH_BINARY)  # THIS IS THE FINAL MASK!!!!
        # Search for contours and select the biggest one
        contours, hierarchy = cv2.findContours(mask, cv2.RECURS_FILTER, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        # Draw the contour on the new mask and perform the bitwise operation
        mask = cv2.drawContours(mask, [cnt], -1, 255, -1)

        path = os.path.join(gt_path)
        gt_filename = str(filename.split('.')[0]) + '_Segmentation.png'
        image_gt = cv2.imread(os.path.join(path, gt_filename), cv2.IMREAD_GRAYSCALE)

        dice_score = dice(mask, image_gt)

        dice_array.append(dice_score)
        dice_mean = np.mean(dice_array)

        temp_metrics = pd.Series([filename, dice_score, dice_mean],
                                 index=['filename', 'dice', 'mean_dice'])
        metrics = metrics.append(temp_metrics, ignore_index=True)

        print(metrics.head(60))
        print("_" * 20)

        plot_results_k(image_rgb,image_gt,mask,dice_score,filename,True)

    if dice_mean > max_mean_dice:
        max_mean_dice = dice_mean
        kernel_best = kernel_size

    print(kernel_best)
# metrics[cols].to_csv(save_dir + "otsu_metrics.csv", index=False)
