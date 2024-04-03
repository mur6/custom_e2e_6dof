from pathlib import Path
import pickle
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
    # load models_pcd.pkl
    with Path("checkpoints/models_pcd.pkl").open("rb") as f:
        models_pcd = pickle.load(f)
    print(type(models_pcd))
    print(f"models_pcd: {models_pcd.shape}")
    train_dataset, val_dataset = get_data()
    for data_dict in train_dataset:
        for key, val in data_dict.items():
            print(f"{key}: {val.shape} {val.dtype}")
        break
    # with Path("models_pcd.pkl").open("wb") as f:
    #     pickle.dump(self.models_pcd, f)


if __name__ == "__main__":
    main()
