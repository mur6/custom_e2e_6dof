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

from utils import PROPSPoseDataset
import utils

# vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# feature_extractor = FeatureExtraction(pretrained_model=vgg16)
# segmentation_branch = SegmentationBranch()
# translation_branch = TranslationBranch()
# rotation_branch = RotationBranch()


def get_data():
    PATH = "."
    train_dataset = PROPSPoseDataset(PATH, "train")
    val_dataset = PROPSPoseDataset(PATH, "val")
    return train_dataset, val_dataset


def main():
    get_data()


if __name__ == "__main__":
    main()
