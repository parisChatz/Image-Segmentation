import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Custom Libs
from my_paths import directory, histograms_path

save_graphs, save_data = False, False

metrics = {'filename': [], 'h': [], 's': [], 'v': []}
cols = ['filename', 'h', 's', 'v']
metrics = pd.DataFrame(data=metrics)

color = ('h', 's', 'v')
bluemax, greenmax, redmax = 0, 0, 0
fig_all = 0
fig_b = 0
fig_g = 0
fig_r = 0

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        if filename == "ISIC_0000019.jpg" or filename == "ISIC_0000095.jpg" or filename == "ISIC_0000214.jpg":
            print(filename)
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            red, green, blue = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

            print('Processing file: ', img_path)
            filtering = True
            if filtering:
                height, width, channels = img.shape
                kernel_size = (15, 15)
                kernel = np.ones(kernel_size, np.uint8)

                # # Perform closing to remove hair and blur the image
                closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
                closing2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

                blur = cv2.blur(closing, kernel_size)
                blur = cv2.medianBlur(blur, kernel_size[0])
                blur = cv2.GaussianBlur(blur, kernel_size, 2)

                img = blur

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # Compute joined hsv histogram
            grid = plt.GridSpec(3, 2)
            plt.figure(figsize=(10, 8))
            plt.subplot(grid[0, 0])  # plot in the first cell
            plt.subplots_adjust(hspace=.5)
            plt.title("Hue")
            plt.hist(np.ndarray.flatten(hue), bins=180)
            plt.subplot(grid[1, 0])  # plot in the second cell
            plt.title("Saturation")
            plt.hist(np.ndarray.flatten(sat), bins=128)

            plt.subplot(grid[2, 0])  # plot in the third cell
            plt.title("Luminosity Value")
            plt.hist(np.ndarray.flatten(val), bins=128)
            plt.subplot(grid[:, 1])  # plot in the 4th cell
            plt.title("RGB_image {}".format(filename))
            plt.imshow(img_rgb)
            plt.show()

            # grid = plt.GridSpec(3, 2)
            # plt.figure(figsize=(10, 8))
            # plt.subplot(grid[0, 0])  # plot in the first cell
            # plt.subplots_adjust(hspace=.5)
            # plt.title("RED")
            # plt.hist(np.ndarray.flatten(red), bins=180)
            # plt.subplot(grid[1, 0])  # plot in the second cell
            # plt.title("GREEN")
            # plt.hist(np.ndarray.flatten(green), bins=128)
            #
            # plt.subplot(grid[2, 0])  # plot in the third cell
            # plt.title("BLUE Value")
            # plt.hist(np.ndarray.flatten(blue), bins=128)
            # plt.subplot(grid[:, 1])  # plot in the 4th cell
            # plt.title("RGB_image {}".format(filename))
            # plt.imshow(img_rgb)
            # plt.show()

            # grid = plt.GridSpec(2, 1)
            # plt.subplot(grid[0, 0])  # plot in the 4th cell
            # plt.title("Histogram")
            # plt.hist(np.ndarray.flatten(img_gray), bins=255)
            # plt.subplot(grid[1, 0])  # plot in the 4th cell
            # plt.title("Image {}".format(filename))
            # plt.imshow(img_gray, cmap='gray')
            # plt.show()



            # plt.show(block=False)
            # plt.pause(4)
            # plt.close()



    else:
        continue

# plt.show(block=False)
# plt.pause(14)
# plt.close()
# print(metrics[cols].head(10))

if save_graphs:
    if os.path.isdir(histograms_path) is False:
        os.mkdir(histograms_path)
    fig_all.savefig(histograms_path + 'joined_hists.png')
    fig_b.savefig(histograms_path + 'blue_hists.png')
    fig_g.savefig(histograms_path + 'green_hists.png')
    fig_r.savefig(histograms_path + 'red_hists.png')

if save_data:
    metrics[cols].to_csv(histograms_path + "metrics.csv", index=False)
