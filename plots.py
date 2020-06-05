import cv2
import matplotlib.pyplot as plt


def plot_results_k(image_rgb, image_gt, mask, dice_score, filename, auto_close):
    # IMAGE WITH MASK PLOT
    new_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    grid = plt.GridSpec(ncols=2, nrows=2)
    plt.figure(figsize=(12, 8))
    plt.subplot(grid[0, 0])
    plt.title(filename + " RGB image")
    plt.imshow(image_rgb)
    plt.subplot(grid[1, 0])
    plt.title(filename + " mask image")
    plt.imshow(mask, cmap='gray')
    plt.subplot(grid[0, 1])
    plt.title(filename + " segmented image")
    plt.imshow(new_img, cmap='gray')
    plt.subplot(grid[1, 1])
    plt.title(" ground truth, dice: {}".format(dice_score))
    plt.imshow(image_gt, cmap='gray')

    # If autoclose is true then wait 1 sec and then close the graph
    if auto_close:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.show()
