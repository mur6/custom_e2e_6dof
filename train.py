import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from timm.scheduler import CosineLRScheduler

from p4_helper import *
from utils import reset_seed
from utils.grad import rel_error
from torch.utils.data import DataLoader
import numpy as np

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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

from utils import PROPSPoseDataset
from utils import BlenderDataset
import utils

utils.reset_seed(0)


def get_data():
    # NOTE: Set `download=True` for the first time when you set up Google Drive folder.
    # Turn it back to `False` later for faster execution in the future.
    # If this hangs, download and place data in your drive manually.
    train_dataset = PROPSPoseDataset(
        PATH,
        "train",
        download=True,  # False
    )
    val_dataset = PROPSPoseDataset(PATH, "val")
    return train_dataset, val_dataset


def get_blender_datasets():
    BPYCV_BASE_DIR = Path("bpycv_6dof_test/data")
    train_dataset = BlenderDataset(BPYCV_BASE_DIR, split="train")
    val_dataset = BlenderDataset(BPYCV_BASE_DIR, split="val")
    return train_dataset, val_dataset


def main(start_checkpoint_path):
    train_dataset, val_dataset = get_blender_datasets()
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    posecnn_model = PoseCNN(
        pretrained_backbone=vgg16,
        models_pcd=torch.tensor(train_dataset.models_pcd).to(
            DEVICE, dtype=torch.float32
        ),
        cam_intrinsic=train_dataset.cam_intrinsic,
    ).to(DEVICE)
    if start_checkpoint_path is not None:
        state_dict = torch.load(start_checkpoint_path)
        posecnn_model.load_state_dict(state_dict=state_dict)
    posecnn_model.train()

    # optimizer = torch.optim.Adam(
    #     posecnn_model.parameters(), lr=0.001, betas=(0.9, 0.999)
    # )
    optimizer = torch.optim.Adam(
        posecnn_model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )

    # scheduler = torch.optim.StepLR(optimizer, step_size=8, gamma=0.75)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    def lr_lambda(epoch):
        if epoch < 4:
            return 1.3
        elif epoch < 8:
            return 1.15
        elif epoch < 70:
            return 1.0
        elif epoch < 125:
            return 0.95
        return 0.999 ** (epoch - 125)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=80,
    #     lr_min=0.0001,
    #     warmup_t=3,
    #     warmup_lr_init=0.0005,
    #     warmup_prefix=True,
    # )

    loss_history = []
    log_period = 5
    _iter = 0
    sche_count = 0
    sche_period = 4

    st_time = time.time()
    for epoch in range(20):
        train_loss = []
        dataloader.dataset.dataset_type = "train"
        for batch in dataloader:
            for item in batch:
                batch[item] = batch[item].to(DEVICE)
            loss_dict = posecnn_model(batch)
            optimizer.zero_grad()
            total_loss = 0
            for loss in loss_dict:
                total_loss += loss_dict[loss]
            # torch.autograd.set_detect_anomaly(True)
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

            if _iter % log_period == 0:
                lr = scheduler.get_last_lr()[0]
                loss_str = f"[Iter {_iter}][lr: {lr}][loss: {total_loss:.3f}]"
                for key, value in loss_dict.items():
                    loss_str += f"[{key}: {value:.3f}]"

                print(loss_str)
                loss_history.append(total_loss.item())
            if _iter % sche_period == 0:
                scheduler.step(sche_count)
                sche_count += 1

            _iter += 1

        print(
            "Time {0}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
                + ", "
                + "Epoch %02d" % epoch
                + ", "
                + "Training finished"
                + f" , with mean training loss {np.array(train_loss).mean()}"
            )
        )

        torch.save(
            posecnn_model.state_dict(),
            os.path.join(PATH, f"posecnn_model_0412_ep_{epoch:02d}.pth"),
        )

    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.savefig("loss_history.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to start training from",
    )
    args = parser.parse_args()
    main(args.start_checkpoint_path)
