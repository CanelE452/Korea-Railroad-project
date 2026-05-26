"""utils.py — Dataset + Augmentation (NVIDIA DOPE fork + 우리 visibility 추가).

CleanVisiiDopeLoader  : 메인 NDDS PNG+JSON dataset. visibility 자동 계산 포함.
crop                  : PIL image crop 헬퍼
AddRandomContrast     : PIL ImageEnhance 대비 augmentation
AddRandomBrightness   : PIL ImageEnhance 밝기 augmentation
AddNoise              : tensor Gaussian noise augmentation
"""
import io
import json
import os

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageEnhance

from utils_loaders import append_dot, loadimages
from utils_belief import CreateBeliefMap, GenerateMapAffinity, VisualizeAffinityMap, VisualizeBeliefMap
from utils_viz import save_image


def crop(img, i, j, h, w):
    """PIL Image crop."""
    return img.crop((j, i, j + w, i + h))


class AddRandomContrast(object):
    """PIL ImageEnhance 기반 random contrast augmentation."""
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Contrast(im)
        return contrast.enhance(np.random.normal(1, self.sigma))


class AddRandomBrightness(object):
    """PIL ImageEnhance 기반 random brightness augmentation."""
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Brightness(im)
        return contrast.enhance(np.random.normal(1, self.sigma))


class AddNoise(object):
    """tensor 에 Gaussian noise 추가, [-1, 1] clamp."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        t = torch.FloatTensor(tensor.size()).normal_(0, self.std)
        t = tensor.add(t)
        t = torch.clamp(t, -1, 1)
        return t


class CleanVisiiDopeLoader(data.Dataset):
    """NDDS PNG + JSON pair 메인 dataset.
    - Albumentations 로 RandomCrop + Rotate + 색조 augmentation 적용
    - belief / affinity / visibility tensor 생성 후 반환
    - visibility 필드 없는 데이터 (예: challenge v1/v2) 도 기본 1 처리
    - S3 bucket 지원 (use_s3=True)
    """

    # Cuboid face definitions (vertex indices)
    _CUBOID_FACES = [
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # rear
        [0, 1, 5, 4],  # top
        [2, 3, 7, 6],  # bottom
        [1, 2, 6, 5],  # left
        [0, 3, 7, 4],  # right
    ]
    # Per-corner adjacent faces (indices into _CUBOID_FACES)
    _CORNER_FACES = [
        [0, 2, 5],  # corner 0: front, top, right
        [0, 2, 4],  # corner 1: front, top, left
        [0, 3, 4],  # corner 2: front, bottom, left
        [0, 3, 5],  # corner 3: front, bottom, right
        [1, 2, 5],  # corner 4: rear, top, right
        [1, 2, 4],  # corner 5: rear, top, left
        [1, 3, 4],  # corner 6: rear, bottom, left
        [1, 3, 5],  # corner 7: rear, bottom, right
    ]

    def __init__(self, path_dataset, objects=None, sigma=1, output_size=400,
                 extensions=["png"], debug=False,
                 use_s3=False, buckets=[], endpoint_url=None):
        self.path_dataset = path_dataset
        self.objects_interest = list(map(str.lower, objects))
        self.sigma = sigma
        self.output_size = output_size
        self.extensions = append_dot(extensions)
        self.debug = debug

        self.imgs = []
        self.s3_buckets = {}
        self.use_s3 = use_s3

        if self.use_s3:
            import boto3
            self.session = boto3.Session()
            self.s3 = self.session.resource(
                service_name="s3", endpoint_url=endpoint_url)
            for bucket_name in buckets:
                try:
                    self.s3_buckets[bucket_name] = self.s3.Bucket(bucket_name)
                except Exception as e:
                    print(f"Error trying to load bucket {bucket_name} for training data:", e)
            for bucket in self.s3_buckets:
                bucket_objects = [str(obj.key) for obj in self.s3_buckets[bucket].objects.all()]
                jsons = set([j for j in bucket_objects if j.endswith(".json")])
                imgs = [img for img in bucket_objects
                        if img.endswith(tuple(self.extensions))]
                for ext in self.extensions:
                    for img in imgs:
                        if img.endswith(ext) and img.replace(ext, ".json") in jsons:
                            self.imgs.append((img, bucket, img.replace(ext, ".json")))
        else:
            for path_look in path_dataset:
                self.imgs += loadimages(path_look, extensions=self.extensions)

        print("Number of Training Images:", len(self.imgs))

        if debug:
            print("Debuging will be save in debug/")
            if os.path.isdir("debug"):
                print('folder debug/ exists')
            else:
                os.mkdir("debug")
                print('created folder debug/')

    def __len__(self):
        return len(self.imgs)

    def _load_raw(self, index):
        """index → (img numpy, data_json, img_name)."""
        if self.use_s3:
            img_key, bucket, json_key = self.imgs[index]
            mem_img = io.BytesIO()
            object_img = self.s3_buckets[bucket].Object(img_key)
            object_img.download_fileobj(mem_img)
            img = np.array(Image.open(mem_img).convert("RGB"))
            object_json = self.s3_buckets[bucket].Object(json_key)
            data_json = json.load(object_json.get()["Body"])
            img_name = img_key[:-3]
        else:
            path_img, img_name, path_json = self.imgs[index]
            img = np.array(Image.open(path_img).convert("RGB"))
            with open(path_json) as f:
                data_json = json.load(f)
        return img, data_json, img_name

    def _collect_keypoints(self, data_json):
        """objects → 9-keypoint 리스트들 (object 별)."""
        all_kps = []
        for obj in data_json["objects"]:
            if (self.objects_interest is not None
                    and obj["class"].lower() not in self.objects_interest):
                continue
            # visibility 필드 없는 데이터셋도 학습 가능하도록 기본 1 처리
            if obj.get("visibility", 1) > 0:
                kps = obj["projected_cuboid"]
                if len(kps) == 8:
                    kps.append(obj["projected_cuboid_centroid"])
            else:
                kps = [[-100, -100]] * 9
            all_kps.append(kps)
        if len(all_kps) == 0:
            all_kps = [[[-100, -100]] * 9]
        return all_kps

    def __getitem__(self, index):
        img, data_json, img_name = self._load_raw(index)
        all_projected_cuboid_keypoints = self._collect_keypoints(data_json)

        # flatten for albumentations
        flatten_projected_cuboid = []
        for obj in all_projected_cuboid_keypoints:
            for p in obj:
                flatten_projected_cuboid.append(p)

        if self.debug:
            img_to_save = Image.fromarray(img)
            draw = ImageDraw.Draw(img_to_save)
            for p in flatten_projected_cuboid:
                draw.ellipse(
                    (int(p[0]) - 2, int(p[1]) - 2, int(p[0]) + 2, int(p[1]) + 2),
                    fill="green")
            img_to_save.save(f"debug/{img_name.replace('.png','_original.png')}")

        # data augmentation (Albumentations)
        transform = A.Compose(
            [
                A.RandomCrop(width=400, height=400),
                A.Rotate(limit=180),
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                     val_shift_limit=30, p=0.5),
                A.RandomGamma(gamma_limit=(60, 140), p=0.3),
                A.GaussNoise(p=0.5),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        transformed = transform(image=img, keypoints=flatten_projected_cuboid)
        img_transformed = transformed["image"]
        flatten_projected_cuboid_transformed = transformed["keypoints"]

        # resize to output_size if needed
        if not self.output_size == 400:
            transform = A.Compose(
                [A.Resize(width=self.output_size, height=self.output_size)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            transformed = transform(
                image=img_transformed, keypoints=flatten_projected_cuboid_transformed)
            img_transformed_output_size = transformed["image"]
            flatten_projected_cuboid_transformed_output_size = transformed["keypoints"]
        else:
            img_transformed_output_size = img_transformed
            flatten_projected_cuboid_transformed_output_size = flatten_projected_cuboid_transformed

        if self.debug:
            img_transformed_saving = Image.fromarray(img_transformed)
            draw = ImageDraw.Draw(img_transformed_saving)
            for p in flatten_projected_cuboid_transformed:
                draw.ellipse(
                    (int(p[0]) - 2, int(p[1]) - 2, int(p[0]) + 2, int(p[1]) + 2),
                    fill="green")
            img_transformed_saving.save(
                f"debug/{img_name.replace('.png','_transformed.png')}")

        # update keypoint structure
        i_all = 0
        for i_obj, obj in enumerate(all_projected_cuboid_keypoints):
            for i_p, _ in enumerate(obj):
                all_projected_cuboid_keypoints[i_obj][i_p] = \
                    flatten_projected_cuboid_transformed_output_size[i_all]
                i_all += 1

        # belief + affinity
        beliefs = CreateBeliefMap(
            size=int(self.output_size),
            pointsBelief=all_projected_cuboid_keypoints,
            sigma=self.sigma, nbpoints=9, save=False,
        )
        beliefs = torch.from_numpy(np.array(beliefs))
        affinities = GenerateMapAffinity(
            size=int(self.output_size), nb_vertex=8,
            pointsInterest=all_projected_cuboid_keypoints,
            objects_centroid=np.array(all_projected_cuboid_keypoints)[:, -1].tolist(),
            scale=1,
        )

        # tensor 변환
        normalize_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_tensor = transforms.Compose([transforms.ToTensor()])
        img_tensor = normalize_tensor(Image.fromarray(img_transformed))
        img_original = to_tensor(img_transformed)

        if self.debug:
            imgs = VisualizeBeliefMap(beliefs)
            save_image(imgs, f"debug/{img_name.replace('.png','_beliefs.png')}",
                       mean=0, std=1, nrow=3, save=True)
            imgs = VisualizeAffinityMap(affinities)
            save_image(imgs, f"debug/{img_name.replace('.png','_affinities.png')}",
                       mean=0, std=1, nrow=3, save=True)

        img_tensor[torch.isnan(img_tensor)] = 0
        affinities[torch.isnan(affinities)] = 0
        beliefs[torch.isnan(beliefs)] = 0
        img_tensor[torch.isinf(img_tensor)] = 0
        affinities[torch.isinf(affinities)] = 0
        beliefs[torch.isinf(beliefs)] = 0

        visibility = self._compute_visibility(data_json, img.shape, self.output_size)

        return {
            "img": img_tensor,
            "affinities": torch.clamp(affinities, -1, 1),
            "beliefs": torch.clamp(beliefs, 0, 1),
            "file_name": img_name,
            "img_original": img_original,
            "visibility": visibility,
        }

    def _compute_visibility(self, data_json, img_shape, output_size):
        """Per-keypoint geometry-derived visibility (3 levels):
        visible(1.0) / self-occluded(0.5) / out-of-frame(0.0).
        Cuboid face normal 의 front-facing 여부로 판정."""
        H, W = img_shape[:2]
        vis = torch.zeros(9, dtype=torch.float32)

        obj = None
        for o in data_json.get("objects", []):
            if self.objects_interest is None or o["class"].lower() in self.objects_interest:
                obj = o
                break

        if obj is None or obj.get("visibility", 0) <= 0:
            return vis

        kps = obj.get("projected_cuboid", [])
        if len(kps) < 8:
            return vis
        centroid = obj.get("projected_cuboid_centroid", [-100, -100])

        cuboid_3d = obj.get("cuboid", None)
        pose_transform = obj.get("pose_transform", None)

        face_visible = [True] * 6
        if cuboid_3d is not None and len(cuboid_3d) >= 8 and pose_transform is not None:
            try:
                pts = np.array(cuboid_3d[:8], dtype=np.float64)
                M = np.array(pose_transform, dtype=np.float64).reshape(4, 4)
                R = M[:3, :3]
                pts_cam = (R @ pts.T).T + M[:3, 3]
                for fi, face in enumerate(self._CUBOID_FACES):
                    p0, p1, p2 = pts_cam[face[0]], pts_cam[face[1]], pts_cam[face[2]]
                    normal = np.cross(p1 - p0, p2 - p0)
                    face_center = pts_cam[face].mean(axis=0)
                    view_dir = -face_center
                    face_visible[fi] = np.dot(normal, view_dir) > 0
            except Exception:
                pass

        for i in range(8):
            x, y = float(kps[i][0]), float(kps[i][1])
            if x < 0 or y < 0 or x >= W or y >= H:
                vis[i] = 0.0
                continue
            any_visible = any(face_visible[fi] for fi in self._CORNER_FACES[i])
            vis[i] = 1.0 if any_visible else 0.5

        cx, cy = float(centroid[0]), float(centroid[1])
        vis[8] = 1.0 if (0 <= cx < W and 0 <= cy < H) else 0.0
        return vis
