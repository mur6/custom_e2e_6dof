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

from utils import PROPSPoseDataset


class BlenderDataset(Dataset):
    def __init__(self, BPYCV_BASE_DIR, *, split):
        self.split = split
        self.image_dir = BPYCV_BASE_DIR / "images"
        self.mask_dir = BPYCV_BASE_DIR / "masks"
        self.info_dir = BPYCV_BASE_DIR / "info"
        self.vector_dir = BPYCV_BASE_DIR / "vectors"

    def __len__(self):
        return 10  # len(self.all_lst)

    def __getitem__(self, idx):
        """
        obj_sample = (rgb_path, depth_path, objs_dict)
        objs_dict = {
            0: {
                cam_R_m2c:
                cam_t_m2c:
                obj_id:
                bbox_visib:
                visiable_mask_path:
            }
            ...
        }

        data_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs',
            'centermaps', []
        }
        """

        def load_as_array(file_p):
            with Image.open(file_p) as im:
                return np.array(im)

        image_file = self.image_dir / f"{idx:04d}.jpg"
        mask_file = self.mask_dir / "masks" / f"mask_{idx:04d}.png"

        # rgb_path, depth_path, objs_dict = self.all_lst[idx]
        data_dict = {}
        rgb = load_as_array(image_file)

        if self.split == "train" and np.random.rand(1) > 1 - self.rgb_aug_prob:
            rgb = chromatic_transform(rgb)
            rgb = add_noise(rgb)
        rgb = rgb.astype(np.float32) / 255
        data_dict["rgb"] = rgb.transpose((2, 0, 1))

        with Image.open(depth_path) as im:
            data_dict["depth"] = np.array(im)[np.newaxis, :]

        ## TODO data-augmentation of depth
        assert len(objs_dict) <= self.max_instance_num
        objs_id = np.zeros(self.max_instance_num, dtype=np.int16)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))
        centermaps = np.zeros(
            (self.max_instance_num, 3, self.resolution[1], self.resolution[0])
        )
        ## test
        img = cv2.imread(rgb_path)

        for idx in objs_dict.keys():
            if len(objs_dict[idx]["bbox_visib"]) > 0:
                ## have visible mask
                objs_id[idx] = self.id2label[objs_dict[idx]["obj_id"]]
                assert objs_id[idx] > 0
                with Image.open(objs_dict[idx]["visible_mask_path"]) as im:
                    label[objs_id[idx]] = np.array(im, dtype=bool)
                ## [x_min, y_min, width, height]
                bbx[idx] = objs_dict[idx]["bbox_visib"]
                RT = np.zeros((4, 4))
                RT[3, 3] = 1
                RT[:3, :3] = objs_dict[idx]["R"]
                RT[:3, [3]] = objs_dict[idx]["T"]
                RT = np.linalg.inv(RT)
                RTs[idx] = RT[:3]
                center_homo = self.cam_intrinsic @ RT[:3, [3]]
                center = center_homo[:2] / center_homo[2]
                x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
                y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
                xv, yv = np.meshgrid(x, y)
                dx, dy = center[0] - xv, center[1] - yv
                distance = np.sqrt(dx**2 + dy**2)
                nx, ny = dx / distance, dy / distance
                Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]
                centermaps[idx] = np.array([nx, ny, Tz])
                ## test
                img = cv2.circle(
                    img,
                    (int(center[0]), int(center[1])),
                    radius=2,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                centers[idx] = np.array([int(center[0]), int(center[1])])
        label[0] = 1 - label[1:].sum(axis=0)
        # Image.fromarray(label[0].astype(np.uint8) * 255).save("testlabel.png")
        # Image.open(rgb_path).save("testrgb.png")
        # cv2.imwrite("testcenter.png", img)
        data_dict["objs_id"] = objs_id
        data_dict["label"] = label
        data_dict["bbx"] = bbx
        data_dict["RTs"] = RTs
        data_dict["centermaps"] = centermaps.reshape(
            -1, self.resolution[1], self.resolution[0]
        )
        data_dict["centers"] = centers
        return data_dict


def get_custom_data():
    BPYCV_BASE_DIR = Path("bpycv_6dof_test/data")
    train_dataset = BlenderDataLoader(BPYCV_BASE_DIR, split="train")
    val_dataset = BlenderDataLoader(BPYCV_BASE_DIR, split="train")
    return train_dataset, val_dataset


def main():
    # load models_pcd.pkl
    with Path("checkpoints/models_pcd.pkl").open("rb") as f:
        models_pcd = pickle.load(f)
    print(f"models_pcd: {models_pcd.shape}")
    print(models_pcd)
    train_dataset, val_dataset = get_custom_data()
    print("===========================")


if __name__ == "__main__":
    main()
