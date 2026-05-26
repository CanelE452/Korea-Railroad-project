"""utils.py — 파일/이미지 탐색 (재귀 탐색, NDDS PNG+JSON pair 매칭).

default_loader      : PIL RGB image 로드
append_dot          : "png" → ".png" 변환 (extensions 정규화)
loadimages          : 폴더 재귀 탐색 → (img, name, json) tuple 리스트
loadweights         : 폴더에서 *.pth 정렬 리스트
loadimages_inference: 재귀 탐색 → image 만 (json 필수 X)
"""
import os
from os.path import exists
from PIL import Image


def default_loader(path):
    return Image.open(path).convert("RGB")


def append_dot(extensions):
    """'png' → '.png' 정규화 (이미 . 있으면 유지)."""
    res = []
    for ext in extensions:
        if not ext.startswith("."):
            res.append(f".{ext}")
        else:
            res.append(ext)
    return res


def loadimages(root, extensions=["png"]):
    """root 재귀 탐색 → PNG+JSON pair 가 있는 image 만 리스트로.

    Returns: list of (imgpath, relative_name, jsonpath).
    """
    imgs = []
    extensions = append_dot(extensions)

    def add_json_files(path):
        for ext in extensions:
            for file in os.listdir(path):
                imgpath = os.path.join(path, file)
                if (imgpath.endswith(ext)
                        and exists(imgpath)
                        and exists(imgpath.replace(ext, ".json"))):
                    imgs.append((
                        imgpath,
                        imgpath.replace(path, "").replace("/", ""),
                        imgpath.replace(ext, ".json"),
                    ))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isdir(os.path.join(path, o))
        ]
        for path_entry in folders:
            explore(path_entry)
        add_json_files(path)

    explore(root)
    return imgs


def loadweights(root):
    """root 가 .pth 파일이면 [root], 폴더면 그 안의 *.pth 정렬 리스트."""
    if root.endswith(".pth") and os.path.isfile(root):
        return [root]
    weights = [
        os.path.join(root, f)
        for f in os.listdir(root)
        if os.path.isfile(os.path.join(root, f)) and f.endswith(".pth")
    ]
    weights.sort()
    return weights


def loadimages_inference(root, extensions):
    """추론용 — JSON 없는 image 도 포함. (imgs, imgsname) 반환."""
    imgs, imgsname = [], []
    extensions = append_dot(extensions)

    def add_imgs(path):
        for ext in extensions:
            for file in os.listdir(path):
                imgpath = os.path.join(path, file)
                if imgpath.endswith(ext) and exists(imgpath):
                    imgs.append(imgpath)
                    imgsname.append(imgpath.replace(root, ""))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isdir(os.path.join(path, o))
        ]
        for path_entry in folders:
            explore(path_entry)
        add_imgs(path)

    explore(root)
    return imgs, imgsname
