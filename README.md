# meditorch

This repo contains useful functions for biomedical datasets - processing, development, and testing. You can find notebooks implementing image segmentation models using PyTorch on Python 3.

# Getting Started

To install PyTorch, see installation instructions on the PyTorch [website](https://pytorch.org/).

To install meditorch:
```
git clone https://github.com/jinglescode/meditorch.git
pip install -r requirements.txt
```

See dependency in [requirements.txt](https://github.com/jinglescode/meditorch/blob/master/requirements.txt).

# Demos

### 1. U-Net for Optic Disc and Cup Segmentation

Glaucoma is an eye disease that occurs without the onset of symptoms at initial, and late diagnosis results in irre- versible degeneration of retinal ganglion cells. As glaucomatous change manifests as structural changes to the optical disk and cup, clinicians rely on the cup to disc ratio to assess glaucomatous damages. This notebook extracts fundus images and annotated optic disc and optic cup, and perform image segmentation using U-Net, Attention U-Net and UNet++. We also visualize segmentations results with various U-Net based models.

Open Drishti-visualize-results.ipynb on [colab](https://colab.research.google.com/github/jinglescode/meditorch/blob/master/demo/Drishti-visualize-results.ipynb).

### 2. U-Net for Optic Disc and Cup Segmentation (compare over 30 runs)

We compare image segmentation performance of U-Net, Attention U-Net and UNet++. Due to the stochastic nature of machine learning, the performance of each method can be affected by the randomness in data shuffle, randomness in weights initialization, and randomness in GPU. To facilitate better comparison, we will train and validate each method with 30 randomly selected seed numbers. The reported performance of each method is the average result of all 30 runs.

Open Drishti-compare-unet-atten_unet-nested_unet-30runs.ipynb on [colab](https://colab.research.google.com/github/jinglescode/meditorch/blob/master/demo/Drishti-compare-unet-atten_unet-nested_unet-30runs.ipynb).

### 3. Nucleus Detection (2018 Data Science Bowl)

Segmenting the nuclei of cells in microscopy images is critical for many biomedical applications; by measuring how cells react to various treatments, researchers can understand the underlying biological processes at work. Therefore, automat- ing nucleus detection could help unlock cures faster. In this notebook, we compare several deep learning architectures for image segmentation. We compared the performance of various U-Net based architectures, DeepLab network, and modified U-Net with various backbone; to perform segmentation of nucleus images from the 2018 Data Science Bowl competition dataset. 

Open Nucleus-compare.ipynb on [colab](https://colab.research.google.com/github/jinglescode/meditorch/blob/master/demo/Nucleus-compare.ipynb).
