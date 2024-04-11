from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


from rob599 import chromatic_transform, add_noise


class BlenderDataset(Dataset):
    def __init__(self, base_dir, *, split):
        self.split = split
        dataset_dir = base_dir / split
        assert dataset_dir.exists(), f"{dataset_dir} does not exist"
        self.data_count = len(list((dataset_dir / "images").glob("*.jpg")))
        self.image_dir = dataset_dir / "images"
        self.mask_dir = dataset_dir / "masks"
        self.depth_dir = dataset_dir / "depth"
        self.info_dir = dataset_dir / "info"
        self.vector_dir = dataset_dir / "vectors"
        ## parameter
        self.max_instance_num = 10
        self.H = 480
        self.W = 640
        self.rgb_aug_prob = 0.4
        # self.cam_intrinsic = np.array(
        #     [[902.19, 0.0, 342.35], [0.0, 902.39, 252.23], [0.0, 0.0, 1.0]]
        # )
        self.cam_intrinsic = np.array(
            [[888.8889, 0.0, 320.0], [0.0, 888.8889, 240.0], [0.0, 0.0, 1.0]]
        )
        self.resolution = [640, 480]
        cube_edge = 0.055
        cube_edge_half = cube_edge / 2.0
        cube_xyz = [
            [cube_edge_half, cube_edge_half, cube_edge_half],
            [cube_edge_half, cube_edge_half, -cube_edge_half],
            [cube_edge_half, -cube_edge_half, cube_edge_half],
            [cube_edge_half, -cube_edge_half, -cube_edge_half],
            [-cube_edge_half, cube_edge_half, cube_edge_half],
            [-cube_edge_half, cube_edge_half, -cube_edge_half],
            [-cube_edge_half, -cube_edge_half, cube_edge_half],
            [-cube_edge_half, -cube_edge_half, -cube_edge_half],
        ]
        cube_coords = np.array(cube_xyz)
        self.models_pcd = np.zeros((10, 8, 3))
        self.models_pcd[0] = cube_coords

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        def load_as_array(file_p):
            with Image.open(file_p) as im:
                return np.array(im)

        image_file = self.image_dir / f"{index:04d}.jpg"
        mask_file = self.mask_dir / f"mask_{index:04d}.png"
        depth_file = self.depth_dir / f"{index:04d}.png"
        info_d = np.load(self.info_dir / f"pose_{index:04d}.npz")

        # print(f"info_d: {info_d}")
        # print(f"info_d.keys(): {info_d.keys()}")
        # print("==========================================")
        # for key in info_d.keys():
        #     print(f"{key}: {info_d[key]}")
        # print("==========================================")

        # rgb_path, depth_path, objs_dict = self.all_lst[idx]
        data_dict = {}
        rgb = load_as_array(image_file)
        if self.split == "train" and np.random.rand(1) > 1 - self.rgb_aug_prob:
            rgb = chromatic_transform(rgb)
            rgb = add_noise(rgb)

        rgb = rgb.astype(np.float32) / 255
        data_dict["rgb"] = rgb.transpose((2, 0, 1))

        with Image.open(depth_file) as im:
            data_dict["depth"] = np.array(im)[np.newaxis, :].astype(np.int32)

        # assert len(objs_dict) <= self.max_instance_num
        objs_id = np.zeros(self.max_instance_num, dtype=np.int32)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))
        centermaps = np.zeros(
            (self.max_instance_num, 3, self.resolution[1], self.resolution[0])
        )

        # for idx in objs_dict.keys():
        #     if len(objs_dict[idx]["bbox_visib"]) > 0:
        idx = 0
        ## have visible mask
        objs_id[idx] = 1  # 多分 0か1 self.id2label[objs_dict[idx]["obj_id"]]

        # assert objs_id[idx] > 0
        mask = load_as_array(mask_file)
        # print(f"mask: shape={mask.shape} dtype={mask.dtype}")
        # print(f"mask: min={mask.min()} max={mask.max()}")

        label[0] = (mask == 0).astype(bool)  # np.array(im, dtype=bool)
        label[1] = mask.astype(bool)
        # print(f"label: shape={label.shape} dtype={label.dtype}")
        # print(f"label: min={label.min()} max={label.max()}")

        bbx[idx] = info_d["bbox_xywh"]
        # print(f"bbx: {bbx}")

        poses = info_d["poses"]
        pose = poses[:, :, idx]
        R, T = pose[:, :3], pose[:, [3]]
        # print(f"R: shape={R.shape} dtype={R.dtype}")
        # print(f"T: shape={T.shape} dtype={T.dtype}")
        RT = np.zeros((4, 4))
        RT[3, 3] = 1
        RT[:3, :3] = R
        RT[:3, [3]] = T
        # RT = np.linalg.inv(RT)
        RTs[idx] = RT[:3]

        cam_intrinsic = info_d["intrinsic_matrix"]
        center_homo = cam_intrinsic @ RT[:3, [3]]
        center = center_homo[:2] / center_homo[2]
        x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
        y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
        xv, yv = np.meshgrid(x, y)
        dx, dy = center[0] - xv, center[1] - yv
        distance = np.sqrt(dx**2 + dy**2)
        nx, ny = dx / distance, dy / distance
        Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]
        centermaps[idx] = np.array([nx, ny, Tz])
        cc = np.squeeze(center, axis=1).astype(np.int32)
        # print(f"cc: {cc}")
        centers[idx] = np.array(cc)
        # print(f"centers[0]: {centers[idx]} dtype={centers[idx].dtype}")
        label[0] = 1 - label[1:].sum(axis=0)

        data_dict["objs_id"] = objs_id
        data_dict["label"] = label
        data_dict["bbx"] = bbx
        data_dict["RTs"] = RTs
        data_dict["centermaps"] = centermaps.reshape(
            -1, self.resolution[1], self.resolution[0]
        )
        data_dict["centers"] = centers
        return data_dict


def get_blender_datasets():
    BPYCV_BASE_DIR = Path("bpycv_6dof_test/data")
    train_dataset = BlenderDataset(BPYCV_BASE_DIR, split="train")
    val_dataset = BlenderDataset(BPYCV_BASE_DIR, split="val")
    return train_dataset, val_dataset
