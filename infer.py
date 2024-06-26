from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision

from p4_helper import *
from utils import reset_seed
from utils.grad import rel_error
import rob599
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

import multiprocessing

NUM_CLASSES = 10
BATCH_SIZE = 4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

from utils import CustomDataset, get_blender_datasets
import utils

utils.reset_seed(0)


def main():
    train_dataset, val_dataset = get_blender_datasets()
    rob599.reset_seed(0)

    dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    posecnn_model = PoseCNN(
        pretrained_backbone=vgg16,
        models_pcd=torch.tensor(val_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
        cam_intrinsic=val_dataset.cam_intrinsic,
    ).to(DEVICE)
    posecnn_model.eval()
    model_ckpt_path = Path("checkpoints") / "posecnn_model_2_ep_09.pth"
    state_dict = torch.load(model_ckpt_path, map_location=DEVICE)
    posecnn_model.load_state_dict(state_dict=state_dict)
    num_samples = 2
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 10))
    axes = axes.flatten()
    for i, data_dict in enumerate(dataloader):
        output_dict, segmentation = posecnn_model(data_dict)
        # print(f"seg: {type(segmentation)}, out: {type(output_dict)}")
        mask = segmentation.cpu().numpy()
        for key, val in output_dict.items():
            print(f"{key}: {val}")
        print(f"mask: {mask.shape}")
        ax = axes[i]
        ax.imshow(mask[0])
        if i == num_samples - 1:
            break
    plt.show()


if __name__ == "__main__":
    main()
