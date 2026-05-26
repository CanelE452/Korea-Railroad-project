"""self_train.py — Dataset 모듈.

SyntheticDataset       : NDDS JSON GT + image + belief/affinity 생성
RealUnlabeledDataset   : pseudo-label 생성용 unlabeled real (image only)
PseudoLabeledDataset   : filter 통과한 pseudo (img + pseudo_belief), strong aug 적용
"""
from __future__ import annotations
import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import CreateBeliefMap, GenerateMapAffinity


class SyntheticDataset(data.Dataset):
    """Loads synthetic data with NDDS-format JSON annotations."""

    def __init__(self, data_dir, object_name="pallet", image_size=448,
                 output_size=56, sigma=2.0):
        self.data_dir = data_dir
        self.object_name = object_name.lower()
        self.image_size = image_size
        self.output_size = output_size
        self.sigma = sigma

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.samples = []
        dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        for d in dirs:
            for ext in ["*.png", "*.jpg"]:
                for img_path in sorted(glob.glob(os.path.join(d, ext))):
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path))
        print(f"SyntheticDataset: {len(self.samples)} samples from {dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, json_path = self.samples[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))

        with open(json_path) as f:
            ann = json.load(f)

        keypoints = self._extract_keypoints(ann, h_orig, w_orig)
        beliefs = self._generate_beliefs(keypoints)
        affinities = self._generate_affinities(keypoints)
        img_tensor = self.normalize(img)

        beliefs[torch.isnan(beliefs) | torch.isinf(beliefs)] = 0
        affinities[torch.isnan(affinities) | torch.isinf(affinities)] = 0

        return {
            "img": img_tensor,
            "beliefs": torch.clamp(beliefs, 0, 1),
            "affinities": torch.clamp(affinities, -1, 1),
            "type": "synthetic",
        }

    def _extract_keypoints(self, ann, h_orig, w_orig):
        """Extract 9 keypoints from NDDS annotation, scaled to image_size."""
        keypoints = [[-100, -100]] * 9
        for obj in ann.get("objects", []):
            if obj.get("class", "").lower() != self.object_name:
                continue
            if obj.get("visibility", 0) <= 0:
                continue
            cuboid = obj.get("projected_cuboid", [])
            centroid = obj.get("projected_cuboid_centroid", [-100, -100])
            if len(cuboid) == 8:
                for i in range(8):
                    keypoints[i] = [
                        cuboid[i][0] * self.image_size / w_orig,
                        cuboid[i][1] * self.image_size / h_orig,
                    ]
                keypoints[8] = [
                    centroid[0] * self.image_size / w_orig,
                    centroid[1] * self.image_size / h_orig,
                ]
            break
        return [keypoints]

    def _generate_beliefs(self, all_keypoints):
        scale = self.output_size / self.image_size
        scaled_kps = [[[x * scale, y * scale] for x, y in kp] for kp in all_keypoints]
        beliefs = CreateBeliefMap(
            size=self.output_size, pointsBelief=scaled_kps,
            sigma=self.sigma, nbpoints=9, save=False,
        )
        return torch.from_numpy(np.array(beliefs)).float()

    def _generate_affinities(self, all_keypoints):
        scale = self.output_size / self.image_size
        scaled_kps = [[[x * scale, y * scale] for x, y in kp] for kp in all_keypoints]
        centroids = [kp[8] for kp in scaled_kps]
        affinities = GenerateMapAffinity(
            size=self.output_size, nb_vertex=8,
            pointsInterest=scaled_kps, objects_centroid=centroids, scale=1,
        )
        return affinities.float()


class RealUnlabeledDataset(data.Dataset):
    """Loads unlabeled real images for pseudo-label generation."""

    def __init__(self, data_dir, image_size=448):
        self.image_size = image_size
        self.image_paths = []
        dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        for d in dirs:
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_paths.extend(sorted(glob.glob(os.path.join(d, ext))))
        print(f"RealUnlabeledDataset: {len(self.image_paths)} images from {dirs}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        # float [0, 1], augmentation 적용 전이라 normalize 안 함
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"img": img_tensor, "path": img_path}


class PseudoLabeledDataset(data.Dataset):
    """Dataset of pseudo-labeled real images accepted by geometric filter."""

    def __init__(self, entries, image_size=448, strong_aug=None):
        """
        Args:
            entries: list of dicts {"img": (C, H, W) float [0,1], "beliefs": (9, H', W')}
            image_size: target image size.
            strong_aug: StrongAugmentation instance (or None).
        """
        self.entries = entries
        self.image_size = image_size
        self.strong_aug = strong_aug
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        img = entry["img"].clone()
        if self.strong_aug is not None:
            img = self.strong_aug(img)
        img = self.normalize(img)
        return {
            "img": img,
            "beliefs": entry["beliefs"],
            "type": "pseudo_real",
        }
