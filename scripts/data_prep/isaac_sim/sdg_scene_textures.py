"""sdg_scene.py — Procedural texture 생성 + 분류.

_generate_procedural_textures : 7 패턴 × 3 변형 = 21 PNG 생성 (floor/wall)
_classify_textures            : realistic(70%) / stylized(30%) 카테고리 분류
_pick_weighted_texture        : 가중 샘플링 (creates weighted random pick)
"""
import os
import numpy as np
from PIL import Image


def _generate_procedural_textures(tex_dir, size=512):
    """시각적으로 확실히 구분되는 7가지 바닥/벽 텍스처 × 3 = 21장 생성.

    패턴:
      gray_concrete / green_epoxy / blue_epoxy / red_brick /
      beige_tile / dark_asphalt / wood_plank
    """
    os.makedirs(tex_dir, exist_ok=True)
    existing = [f for f in os.listdir(tex_dir) if f.endswith(".png")]
    expected_count = 7 * 3
    marker = os.path.join(tex_dir, "_v10_marker")
    if os.path.exists(marker) and len(existing) >= expected_count:
        paths = [os.path.join(tex_dir, f)
                 for f in sorted(existing) if f.endswith(".png")]
        print(f"  [TEX] Reusing {len(paths)} v10 procedural textures from {tex_dir}")
        return paths

    for f in existing:
        os.remove(os.path.join(tex_dir, f))

    rng = np.random.RandomState(42)
    paths = []

    def _add_noise(img, rng, large_std=15, fine_std=8, size=512):
        noise_l = rng.normal(0, large_std, (size // 8, size // 8))
        noise_l = np.repeat(np.repeat(noise_l, 8, axis=0), 8, axis=1)[:size, :size]
        noise_f = rng.normal(0, fine_std, (size, size))
        return img + noise_l + noise_f

    for idx in range(3):
        # 1) gray_concrete
        base_rgb = np.array([140 + idx * 15, 138 + idx * 15, 135 + idx * 15], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 18, 10, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"gray_concrete_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 2) green_epoxy
        g_base = rng.randint(100, 140)
        base_rgb = np.array([g_base * 0.45, g_base, g_base * 0.5], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 12, 6, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"green_epoxy_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 3) blue_epoxy
        b_base = rng.randint(110, 150)
        base_rgb = np.array([b_base * 0.4, b_base * 0.55, b_base], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 12, 6, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"blue_epoxy_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 4) red_brick (벽돌 패턴)
        brick_h = rng.choice([32, 48, 64])
        brick_w = brick_h * 2
        r_base, g_base_b, b_base_b = 160 + idx * 20, 85 + idx * 10, 60 + idx * 10
        img = np.full((size, size, 3), [r_base, g_base_b, b_base_b], dtype=np.float32)
        mortar = np.array([180, 175, 165], dtype=np.float32)
        for row in range(0, size, brick_h):
            img[row:min(row + 3, size), :, :] = mortar
            offset = (brick_w // 2) if ((row // brick_h) % 2) else 0
            for col in range(offset, size, brick_w):
                img[row:row + brick_h, max(col - 1, 0):min(col + 2, size), :] = mortar
        for c in range(3):
            img[:, :, c] += rng.normal(0, 8, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"red_brick_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 5) beige_tile (타일 격자)
        tile_sz = rng.choice([64, 128])
        t_base = np.array([210 - idx * 15, 195 - idx * 10, 170 - idx * 10], dtype=np.float32)
        img = np.full((size, size, 3), t_base, dtype=np.float32)
        grout = t_base * 0.6
        for i in range(0, size, tile_sz):
            img[max(i - 1, 0):i + 2, :, :] = grout
            img[:, max(i - 1, 0):i + 2, :] = grout
        for c in range(3):
            img[:, :, c] += rng.normal(0, 5, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"beige_tile_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 6) dark_asphalt
        a_base = 65 + idx * 15
        base_rgb = np.array([a_base, a_base - 3, a_base - 5], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 20, 12, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"dark_asphalt_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 7) wood_plank (목재 판자 + 나뭇결)
        w_base = np.array([155 + idx * 15, 120 + idx * 10, 75 + idx * 8], dtype=np.float32)
        img = np.full((size, size, 3), w_base, dtype=np.float32)
        plank_w = rng.choice([48, 64, 96])
        for col in range(0, size, plank_w):
            plank_tint = rng.uniform(0.85, 1.15)
            img[:, col:col + plank_w, :] *= plank_tint
            img[:, max(col - 1, 0):col + 1, :] *= 0.6
        for _ in range(rng.randint(15, 30)):
            y = rng.randint(0, size)
            thickness = rng.randint(1, 4)
            img[y:y + thickness, :, :] *= rng.uniform(0.88, 0.96)
        for c in range(3):
            img[:, :, c] += rng.normal(0, 6, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"wood_plank_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

    with open(marker, "w") as f:
        f.write("v10")

    print(f"  [TEX] Generated {len(paths)} diverse procedural textures in {tex_dir}")
    return paths


def _classify_textures(texture_paths):
    """현실적(70%) / stylized(30%) 분류 — 가중 샘플링용.

    현실적: gray_concrete, green_epoxy, blue_epoxy, dark_asphalt
    stylized: red_brick, beige_tile, wood_plank
    """
    realistic = []
    stylized = []
    for p in texture_paths:
        basename = os.path.basename(p).lower()
        if any(k in basename for k in
               ("gray_concrete", "green_epoxy", "blue_epoxy", "dark_asphalt")):
            realistic.append(p)
        else:
            stylized.append(p)
    return realistic, stylized


def _pick_weighted_texture(rng, realistic, stylized, realistic_prob=0.7):
    """realistic_prob 확률로 realistic, 나머지는 stylized 샘플링."""
    if not stylized or float(rng.random()) < realistic_prob:
        return realistic[int(rng.integers(len(realistic)))]
    return stylized[int(rng.integers(len(stylized)))]
