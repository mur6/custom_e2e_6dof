from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

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
    print("===========================")
    for data_dict in train_dataset:
        for key, val in data_dict.items():
            print(f"{key}: {val.shape} {val.dtype}")
        break
    print("===========================")
    rgb: np.ndarray = data_dict["rgb"]
    objs_id = data_dict["objs_id"]
    label = data_dict["label"]  # label: (11, 480, 640) bool

    print(f"objs_id: {objs_id}")  # objs_id: (10,) int16

    # bbx: (10, 4) float64
    # RTs: (10, 3, 4) float64
    # centermaps: (30, 480, 640) float64
    centers = data_dict["centers"]

    depth = data_dict["depth"]
    fig, axes = plt.subplots(2, 2)
    ax = axes.flatten()
    ax[0].imshow(rgb.transpose(1, 2, 0))
    # plot center
    ax[0].scatter(centers[:, 0], centers[:, 1], c="r", s=10)
    ax[0].set_title("rgb")
    ax[1].imshow(np.squeeze(depth), cmap="gray")
    ax[1].set_title("depth")
    ax[2].imshow(label[0], cmap="gray")
    ax[2].set_title("label[0]: background")
    ax[3].imshow(label[1], cmap="gray")
    ax[3].set_title("label[1]: object")
    plt.show()
    # objs_id = data_dict["objs_id"]
    # print(f"objs_id: {objs_id}")
    # bbx = data_dict["bbx"]
    # print(f"bbx: {bbx}")


if __name__ == "__main__":
    main()
