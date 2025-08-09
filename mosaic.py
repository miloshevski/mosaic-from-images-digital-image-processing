import os
import cv2
import numpy as np
from glob import glob

def average_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Пресметува просек на боја во LAB простор."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)

def load_tiles(tiles_dir: str, tile_size: int):
    """Вчитува плочки од папката, автоматски ги прилагодува на квадрат."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(glob(os.path.join(tiles_dir, ext)))
    if not paths:
        raise FileNotFoundError("No tile images found in tiles_dir")

    tiles_small = []
    tiles_lab = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue

        # Кроп до квадрат
        h, w = img.shape[:2]
        min_side = min(h, w)
        y0 = (h - min_side) // 2
        x0 = (w - min_side) // 2
        img_cropped = img[y0:y0+min_side, x0:x0+min_side]

        # Ресајз на големина на плочка
        img_resized = cv2.resize(img_cropped, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        tiles_small.append(img_resized)
        tiles_lab.append(average_lab(img_resized))

    if len(tiles_small) == 0:
        raise RuntimeError("Failed to load any valid tile images.")

    return np.array(tiles_small), np.array(tiles_lab, dtype=np.float32)

def build_mosaic(target_path: str, tiles_dir: str, tile_size: int = 16, blend: float = 0.15, no_immediate_repeat: bool = True, max_width: int = 800):
    """Гради мозаиќ од целната слика и плочките."""
    # 1) Вчитај целна слика (оригинална)
    target_original = cv2.imread(target_path)
    if target_original is None:
        raise FileNotFoundError(f"Can't read target image: {target_path}")

    # Намали ја ако е поширока од max_width
    target = target_original.copy()
    if target.shape[1] > max_width:
        scale = max_width / target.shape[1]
        new_size = (max_width, int(target.shape[0] * scale))
        target = cv2.resize(target, new_size, interpolation=cv2.INTER_AREA)

    h, w = target.shape[:2]
    # За да се поклопи мрежата со tile_size
    h_new = (h // tile_size) * tile_size
    w_new = (w // tile_size) * tile_size
    target = cv2.resize(target, (w_new, h_new), interpolation=cv2.INTER_AREA)

    # 2) Вчитај плочки
    tiles_small, tiles_lab = load_tiles(tiles_dir, tile_size)

    # 3) Подготви празно платно
    mosaic = np.zeros_like(target)

    # 4) LAB за целната
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # 5) Гради мозаиќ
    rows = h_new // tile_size
    cols = w_new // tile_size
    last_used_idx = -1

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size

            patch_lab = target_lab[y0:y1, x0:x1]
            mean_lab = patch_lab.reshape(-1, 3).mean(axis=0)

            dists = np.linalg.norm(tiles_lab - mean_lab, axis=1)

            if no_immediate_repeat and last_used_idx >= 0:
                dists[last_used_idx] += 5.0

            idx = int(np.argmin(dists))
            tile_img = tiles_small[idx]

            mosaic[y0:y1, x0:x1] = tile_img
            last_used_idx = idx

    if blend > 0:
        mosaic = cv2.addWeighted(mosaic, 1 - blend, target, blend, 0)

    return mosaic, target_original  # враќаме финалниот мозаиќ и вистинската оригинална слика

if __name__ == "__main__":
    # Поставки
    TARGET = "target.jpg"   # Целната слика
    TILES_DIR = "tiles"     # Папка со плочки
    TILE_SIZE = 16          # Помало = повеќе детали
    BLEND = 0.12            # Степен на блендање
    MAX_WIDTH = 800         # Максимална ширина на target слика

    out, original_image = build_mosaic(TARGET, TILES_DIR, TILE_SIZE, BLEND, max_width=MAX_WIDTH)

    cv2.imshow("Original Image", original_image)
    cv2.imshow("Mosaic", out)
    cv2.imwrite("mosaic_output.jpg", out)
    print("✅ Saved mosaic_output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
