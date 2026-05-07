"""Photometric augmentations for FixMatch-based self-training.

Provides weak and strong augmentation transforms for images.
Only photometric transforms are used (no geometric transforms)
to avoid the need for keypoint coordinate transformation.

Usage:
    weak_aug = WeakAugmentation(brightness=0.1, contrast=0.1)
    strong_aug = StrongAugmentation(cfg)
    img_weak = weak_aug(img)
    img_strong = strong_aug(img)
"""

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class WeakAugmentation:
    """Minimal photometric augmentation for pseudo-label generation.

    Applied to images before feeding to the model in eval mode
    to generate pseudo-labels. Should be mild enough to preserve
    detection accuracy while adding slight variation.
    """

    def __init__(self, brightness=0.1, contrast=0.1, noise_std=0.01):
        """
        Args:
            brightness: max brightness adjustment (+/-).
            contrast: max contrast adjustment (+/-).
            noise_std: standard deviation of additive Gaussian noise.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, img):
        """Apply weak augmentation.

        Args:
            img: torch.Tensor of shape (C, H, W) in [0, 1] range,
                 or np.ndarray of shape (H, W, C) in [0, 255] range.

        Returns:
            Augmented image in the same format as input.
        """
        if isinstance(img, np.ndarray):
            return self._apply_numpy(img)
        return self._apply_tensor(img)

    def _apply_tensor(self, img):
        # Brightness
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = TF.adjust_brightness(img, factor)

        # Contrast
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        img = TF.adjust_contrast(img, factor)

        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)

        return img

    def _apply_numpy(self, img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
            was_uint8 = True
        else:
            was_uint8 = False

        # Brightness
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * factor, 0.0, 1.0)

        # Contrast
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
            img = np.clip(img + noise, 0.0, 1.0)

        if was_uint8:
            img = (img * 255).astype(np.uint8)

        return img


class StrongAugmentation:
    """Aggressive photometric augmentation for self-training.

    Applied to real unlabeled images during training (with pseudo-labels).
    The model must learn to produce consistent predictions despite
    strong appearance changes.
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict with augmentation parameters. Uses defaults if None.
        """
        cfg = config or {}
        self.brightness = cfg.get("brightness", 0.4)
        self.contrast = cfg.get("contrast", 0.4)
        self.saturation = cfg.get("saturation", 0.4)
        self.hue = cfg.get("hue", 0.1)
        self.erasing_prob = cfg.get("random_erasing_prob", 0.5)
        self.erasing_scale = tuple(cfg.get("random_erasing_scale", [0.02, 0.2]))
        self.blur_prob = cfg.get("gaussian_blur_prob", 0.5)
        self.blur_kernel = cfg.get("gaussian_blur_kernel", [3, 7])
        self.noise_std = cfg.get("gaussian_noise_std", 0.05)

        # Build torchvision transforms
        self.color_jitter = T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )
        self.random_erasing = T.RandomErasing(
            p=self.erasing_prob,
            scale=self.erasing_scale,
            ratio=(0.3, 3.3),
            value=0,
        )

    def __call__(self, img):
        """Apply strong augmentation.

        Args:
            img: torch.Tensor of shape (C, H, W) in [0, 1] range.

        Returns:
            Augmented tensor of same shape.
        """
        if isinstance(img, np.ndarray):
            img = self._numpy_to_tensor(img)

        # Color jitter (brightness, contrast, saturation, hue)
        img = self.color_jitter(img)

        # Gaussian blur
        if random.random() < self.blur_prob:
            kernel_size = random.choice(
                range(self.blur_kernel[0], self.blur_kernel[1] + 1, 2)
            )
            img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])

        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)

        # Random erasing (after converting to tensor range)
        img = self.random_erasing(img)

        return img

    @staticmethod
    def _numpy_to_tensor(img):
        """Convert (H, W, C) uint8 numpy array to (C, H, W) float tensor."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1))
