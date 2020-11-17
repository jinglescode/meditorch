import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


def metrics_line(data):
    phases = list(data.keys())
    metrics = list(data[phases[0]][0].keys())

    i = 0
    fig, axs = plt.subplots(1, len(metrics))
    fig.set_figheight(4)
    fig.set_figwidth(4*len(metrics))
    for metric in metrics:
        for phase in phases:
            axs[i].plot([i[metric] for i in data[phase]], label=phase)
        axs[i].set_title(metric)
        i+=1

    plt.legend()
    plt.show()


def normalise_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

def plot_image_truemask_predictedmask(images, labels, preds):

    input_images_rgb = [reverse_transform(x) for x in images]
    target_masks_rgb = [masks_to_coloredmasks(x) for x in labels]
    pred_rgb = [masks_to_coloredmasks(x) for x in preds]

    img_arrays = [input_images_rgb, target_masks_rgb, pred_rgb]
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

def apply_mask_color(mask, mask_color):
    colored_mask = np.concatenate(([mask[ ... , np.newaxis] * color for color in mask_color]), axis=2)
    return colored_mask.astype(np.uint8)

def masks_to_coloredmasks(mask, normalise=True, colors=None):
    segments_colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])
    if colors is not None:
        segments_colors = colors

    if normalise:
        normalise_mask(mask)

    mask_colored = np.concatenate( [ [apply_mask_color(mask[i], segments_colors[i])] for i in range(len(mask)) ] )
    mask_colored = np.max(mask_colored, axis=0)

    mask_colored = np.where(mask_colored.any(-1,keepdims=True),mask_colored,255)

    return mask_colored

