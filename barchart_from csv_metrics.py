import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import imutils

from my_paths import directory, save_dir

name_list = []
number = []
scores = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        names = filename.split('_')[1].split(".")[0][-3:]
        name_list.append(names)
y_pos = np.arange(len(name_list))

df = pd.read_csv(save_dir + "otsu_metrics.csv")
# print(df.head())

for i, key in enumerate(name_list):
    scores.append(df["dice"][i])
    number.append(i)

plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, name_list, rotation=75)
plt.axhline(0.83, color='#999999')
plt.ylabel('Dice Similarity Score')
plt.title('Bar-chart of Dice score and pictures with 0.83 mean dice score')

plt.show()