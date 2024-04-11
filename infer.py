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
    Path()
    posecnn_model.load_state_dict(torch.load("posecnn_model.pth"))
    num_samples = 5
    for i in range(num_samples):
        out = eval(posecnn_model, dataloader, DEVICE)

        plt.axis("off")
        plt.imshow(out)
        plt.show()


if __name__ == "__main__":
    main()
