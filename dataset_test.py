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

from pose_cnn import (
    PoseCNN,
    FeatureExtraction,
    SegmentationBranch,
    TranslationBranch,
    RotationBranch,
)

# vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# feature_extractor = FeatureExtraction(pretrained_model=vgg16)
# segmentation_branch = SegmentationBranch()
# translation_branch = TranslationBranch()
# rotation_branch = RotationBranch()

def main():
    pass

if __name__ == "__main__":
    main()

