import os
import glob
import numpy as np
from PIL import Image
from skimage.transform import resize

class UtilsImageException(Exception):
    pass

def load_image(path):
    # returns an image of dtype int in range [0, 255]
    return np.asarray(Image.open(path))

def load_set(folder, shuffle=False):
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    data = []
    filenames = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
        filenames.append(img_fn)
    return data, filenames

def resize_image_to_square(img, side, pad_cval=0, dtype=np.float64):
    """Resizes img of shape (h, w, ch) or (h, w) to square of size (side, side, ch)
    or (side, side), respectively, while preserving aspect ratio.
    Image is being padded with pad_cval if needed."""

    if len(img.shape) == 2:
        h, w = img.shape
        if h == w:
            padded = img.copy()
        elif h > w:
            padded = np.full((h, h), pad_cval, dtype=dtype)
            l = int(h / 2 - w / 2)  # guaranteed to be non-negative
            r = l + w
            padded[:, l:r] = img.copy()
        else:
            padded = np.full((w, w), pad_cval, dtype=dtype)
            l = int(w / 2 - h / 2)  # guaranteed to be non-negative
            r = l + h
            padded[l:r, :] = img.copy()
    elif len(img.shape) == 3:
        h, w, ch = img.shape
        if h == w:
            padded = img.copy()
        elif h > w:
            padded = np.full((h, h, ch), pad_cval, dtype=dtype)
            l = int(h / 2 - w / 2)   # guaranteed to be non-negative
            r = l + w
            padded[:, l:r, :] = img.copy()
        else:
            padded = np.full((w, w, ch), pad_cval, dtype=dtype)
            l = int(w / 2 - h / 2)   # guaranteed to be non-negative
            r = l + h
            padded[l:r, :, :] = img.copy()
    else:
        raise UtilsImageException('only images of 2d and 3d shape are accepted')

    resized_img = resize(padded, output_shape=(side, side))

    return resized_img
