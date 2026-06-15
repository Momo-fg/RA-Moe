import numpy as np
import cv2

# =============================
# 5 大区域与 68 点关键点的映射表
# =============================

REGION_LANDMARK_MAPPING_68 = {
    # 眉毛区域（包含眉头内外和中部）
    "brow": list(range(17, 27)),  # 17~26 是两边眉毛 + 27 中点
    # 眼睛区域（上下眼睑 + 眼角）
    "eye": list(range(36, 48)),  # 36~41 左眼，42~47 右眼
    # 鼻子区域（鼻梁 + 鼻翼）
    "nose": list(range(27, 36)),  # 27~30 鼻梁，31~35 鼻翼
    # 嘴巴区域（上下唇 + 嘴角）
    "mouth": list(range(48, 68)),  # 48~59 外轮廓，60~67 内轮廓
    # 下巴区域（下巴点）
    "chin": list(range(6, 13))    # 6~12 下颌线中下部区域
}


def generate_region_mask(
    landmarks: np.ndarray,
    active_regions: list,
    image_size=(224, 224),
    region_landmark_mapping=REGION_LANDMARK_MAPPING_68,
    expand: int = 5
) -> np.ndarray:
    """
    根据活动的 面部区域 列表，使用对应的 landmarks，生成 mask。

    参数：
    - landmarks: np.ndarray, [68, 2] 关键点坐标
    - active_regions: list[str], 如 ['眉毛', '眼睛']
    - image_size: tuple, (高, 宽)
    - region_landmark_mapping: dict, 区域名称 -> landmark 索引
    - expand: int, 扩展边界的像素数

    返回：
    - mask: np.ndarray, 二值图，关键区域为 1，其余为 0
    """
    H, W = image_size
    mask = np.zeros((H, W), np.uint8)

    for region in active_regions:
        if region not in region_landmark_mapping:
            print(f"Warning: 未定义的区域 {region}")
            continue

        pts = []
        for idx in region_landmark_mapping[region]:
            if idx >= len(landmarks):
                continue
            x, y = landmarks[idx]
            # 如果 landmarks 是归一化的（0~1）
            if x <= 1.1 and y <= 1.1:
                x, y = x * W, y * H
            pts.append([int(x), int(y)])

        if len(pts) < 2:
            continue

        pts = np.array(pts, dtype=int)
        x1, y1 = pts.min(axis=0) - expand
        x2, y2 = pts.max(axis=0) + expand
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W - 1), min(y2, H - 1)

        mask[y1:y2 + 1, x1:x2 + 1] = 1

    return mask


if __name__ == "__main__":
    print("generate_region_mask 模块仅用于导入使用，不应直接执行。")
