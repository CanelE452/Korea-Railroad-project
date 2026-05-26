"""utils.py — 시각화 모듈 (NVIDIA DOPE 원본).

make_grid       : tensor → 격자 image grid (torchvision 의 원본 fork)
save_image      : tensor 를 image 파일로 저장 (mean/std denormalize)
Draw            : PIL ImageDraw wrapper (cuboid wireframe + dot + text)
get_image_grid  : tensor grid + denormalize 헬퍼
"""
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

from cuboid import CuboidVertexType, CuboidLineIndexes


def make_grid(tensor, nrow=8, padding=2, normalize=False,
              value_range=None, scale_each=False, pad_value=0):
    """torchvision make_grid 원본 fork — 4D 텐서를 격자 image 로.

    NOTE: 이전 버전은 parameter 이름이 `range` 였는데 이게 built-in `range` 를 shadow 해서
    line 68 `for y in range(ymaps)` 가 TypeError 발생. `value_range` 로 rename.
    """
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)
    if tensor.dim() == 2:
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified."

        def norm_ip(img, mn, mx):
            img.clamp_(min=mn, max=mx)
            img.add_(-mn).div_(mx - mn + 1e-5)

        def norm_range(t, r):
            if r is not None:
                norm_ip(t, r[0], r[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding,
                      width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2, mean=None, std=None, save=True):
    """tensor → image 파일 저장. mean/std 주어지면 denormalize."""
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10, pad_value=1)
    if not mean is None:
        ndarr = (grid.mul(std).add(mean).mul(255).byte()
                 .transpose(0, 2).transpose(0, 1).numpy())
    else:
        ndarr = (grid.mul(0.5).add(0.5).mul(255).byte()
                 .transpose(0, 2).transpose(0, 1).numpy())
    im = Image.fromarray(ndarr)
    if save is True:
        im.save(filename)
    return im, grid


class Draw(object):
    """PIL ImageDraw wrapper — DOPE 추론 결과 시각화 (cuboid wireframe + X + center)."""

    def __init__(self, im):
        self.draw = ImageDraw.Draw(im)
        self.width = im.size[0]

    def draw_line(self, point1, point2, line_color, line_width=2):
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_rectangle(self, point1, point2, line_color=(0, 255, 0), line_width=2):
        self.draw.rectangle([point1, point2], outline=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        if point is not None:
            xy = [
                point[0] - point_radius, point[1] - point_radius,
                point[0] + point_radius, point[1] + point_radius,
            ]
            self.draw.ellipse(xy, fill=point_color, outline=point_color)

    def draw_text(self, point, text, text_color):
        if point is not None:
            self.draw.text(point, text, fill=text_color)

    def draw_cube(self, points, color=(0, 255, 0)):
        """cuboid wireframe + top face X + center dot + 0~8 label."""
        for l in CuboidLineIndexes:
            self.draw_line(points[l[0]], points[l[1]], color, line_width=2)
        X_Indexes = [
            [CuboidVertexType.FrontTopRight, CuboidVertexType.RearTopLeft],
            [CuboidVertexType.FrontTopLeft, CuboidVertexType.RearTopRight],
        ]
        for x in X_Indexes:
            self.draw_line(points[x[0]], points[x[1]], color, line_width=2)
        self.draw_dot(points[CuboidVertexType.Center],
                      point_color=color, point_radius=6)
        for i in range(9):
            self.draw_text(points[i], str(i), (255, 0, 0))


def get_image_grid(tensor, nrow=3, padding=2, mean=None, std=None):
    """tensor grid + denormalize 헬퍼 (save_image 의 in-memory 변형)."""
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=1)
    if not mean is None:
        ndarr = (grid.mul(std).add(mean).mul(255).byte()
                 .transpose(0, 2).transpose(0, 1).numpy())
    else:
        ndarr = (grid.mul(0.5).add(0.5).mul(255).byte()
                 .transpose(0, 2).transpose(0, 1).numpy())
    return Image.fromarray(ndarr)
