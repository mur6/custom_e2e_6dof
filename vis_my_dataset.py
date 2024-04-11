from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

# from utils import reset_seed
# from utils.grad import rel_error
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pose_cnn import (
    PoseCNN,
    FeatureExtraction,
    SegmentationBranch,
    TranslationBranch,
    RotationBranch,
)

from utils import BlenderDataset


def get_blender_datasets():
    BPYCV_BASE_DIR = Path("bpycv_6dof_test/data")
    train_dataset = BlenderDataset(BPYCV_BASE_DIR, split="train")
    val_dataset = BlenderDataset(BPYCV_BASE_DIR, split="train")
    return train_dataset, val_dataset


def visualize_dataset(data_dict):
    rgb = data_dict["rgb"]
    print(f"rgb: shape={rgb.shape} dtype={rgb.dtype}")
    print(f"rgb: min={rgb.min()} max={rgb.max()}")
    objs_id = data_dict["objs_id"]
    label = data_dict["label"]  # label: (11, 480, 640) bool

    print(f"objs_id: {objs_id}")  # objs_id: (10,) int16

    # bbx: (10, 4) float64
    bbx = data_dict["bbx"]
    bbx0 = bbx[0]
    # RTs: (10, 3, 4) float64
    RTs = data_dict["RTs"]
    RTs0 = RTs[0]
    print(f"bbx0: {bbx0}")
    print(f"RTs0: {RTs0}")
    # centermaps: (30, 480, 640) float64
    cetermaps = data_dict["centermaps"]
    print(f"cetermaps: shape={cetermaps.shape} dtype={cetermaps.dtype}")
    print(f"cetermaps: min={cetermaps.min()} max={cetermaps.max()}")
    centers = data_dict["centers"]

    depth = data_dict["depth"]
    print(f"depth: shape={depth.shape} dtype={depth.dtype}")
    print(f"depth: min={depth.min()} max={depth.max()}")
    fig, axes = plt.subplots(2, 3)
    ax = axes.flatten()
    ax[0].imshow(rgb.transpose(1, 2, 0))
    # plot center
    ax[0].scatter(centers[:, 0], centers[:, 1], c="r", s=10)
    ax[0].set_title("rgb")
    depth = (depth - depth.min()).astype(np.float32)
    depth = (depth / depth.max()).astype(np.float32)
    im = ax[1].imshow(np.squeeze(depth), cmap="gray")
    ax[1].set_title("depth")
    plt.colorbar(im, ax=ax[1])
    ax[2].imshow(label[0], cmap="gray")
    ax[2].set_title("label[0]: background")
    ax[3].imshow(label[1], cmap="gray")
    ax[3].set_title("label[1]: object")
    ax[4].imshow(cetermaps[0:3, :, :].transpose(1, 2, 0), cmap="gray")
    ax[4].set_title("centermaps[0:3]")
    ax[5].imshow(rgb.transpose(1, 2, 0))
    for i in range(len(bbx)):
        x, y, w, h = bbx[i]
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="r", linewidth=1)
        ax[5].add_patch(rect)
    ax[5].set_title("rgb with bbox")
    plt.show()


def main():
    train_dataset, val_dataset = get_blender_datasets()
    models_pcd = train_dataset.models_pcd
    print(f"train_dataset.models_pcd: shape={train_dataset.models_pcd}")
    for i, data_dict in enumerate(train_dataset):
        print(f"i: {i}")
        if i == 0:
            break
    print("===========================")
    # visualize_dataset(data_dict)


if __name__ == "__main__":
    main()
