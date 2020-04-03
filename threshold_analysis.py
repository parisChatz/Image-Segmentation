import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Custom Libs
from my_paths import directory, histograms_path

save_graphs, save_data = False, False

metrics = {'filename': [], 'redmax': [], 'greenmax': [], 'bluemax': []}
cols = ['filename', 'redmax', 'greenmax', 'bluemax']
metrics = pd.DataFrame(data=metrics)

color = ('h', 's', 'v')
bluemax, greenmax, redmax = 0, 0, 0
fig_all = 0
fig_b = 0
fig_g = 0
fig_r = 0

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        # todo make them hsv first
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        print('Processing file: ', img_path)

        for i, col in enumerate(color):

            # Compute joined rgb histogram
            fig_all = plt.figure(0)
            histr = cv2.calcHist([hsv], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
            plt.ylim([0, 500000])

            # Bin of max histogram value
            # elem = np.argmax(histr)

            # if col == 'b':
            #     bluemax = elem
            #     hist_b = cv2.calcHist([img], [i], None, [256], [0, 256])
            #     fig_b = plt.figure(1)
            #     plt.plot(hist_b, color=col)
            #     plt.xlim([0, 256])
            #     plt.ylim([0, 500000])
            # elif col == 'g':
            #     greenmax = elem
            #     hist_g = cv2.calcHist([img], [i], None, [256], [0, 256])
            #     fig_g = plt.figure(2)
            #     plt.plot(hist_g, color=col)
            #     plt.xlim([0, 256])
            #     plt.ylim([0, 500000])
            # elif col == 'r':
            #     redmax = elem
            #     hist_r = cv2.calcHist([img], [i], None, [256], [0, 256])
            #     fig_r = plt.figure(3)
            #     plt.plot(hist_r, color=col)
            #     plt.xlim([0, 256])
            #     plt.ylim([0, 500000])
        plt.figure(10)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show(block=False)
        plt.pause(4)
        plt.close()

        # temp_metrics = pd.Series([filename, bluemax, greenmax, redmax],
        #                          index=['filename', 'bluemax', 'greenmax', 'redmax'])
        # metrics = metrics.append(temp_metrics, ignore_index=True)

    else:
        continue

plt.show(block=False)
plt.pause(14)
plt.close()
print(metrics[cols].head(10))

if save_graphs:
    if os.path.isdir(histograms_path) is False:
        os.mkdir(histograms_path)
    fig_all.savefig(histograms_path + 'joined_hists.png')
    fig_b.savefig(histograms_path + 'blue_hists.png')
    fig_g.savefig(histograms_path + 'green_hists.png')
    fig_r.savefig(histograms_path + 'red_hists.png')

if save_data:
    metrics[cols].to_csv(histograms_path + "metrics.csv", index=False)
