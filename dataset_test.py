import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision

from p4_helper import *
from utils import reset_seed
from utils.grad import rel_error
from torch.utils.data import DataLoader

import torchvision.models as models

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
from pose_cnn import (
    PoseCNN,
    FeatureExtraction,
    SegmentationBranch,
    TranslationBranch,
    RotationBranch,
)

feature_extractor = FeatureExtraction(pretrained_model=vgg16)
segmentation_branch = SegmentationBranch()
translation_branch = TranslationBranch()
rotation_branch = RotationBranch()

# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

import multiprocessing

# Set a few constants related to data loading.
NUM_CLASSES = 10
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count()
path = os.getcwd()
PATH = os.path.join(path)
