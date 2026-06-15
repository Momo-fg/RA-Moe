import numpy as np
import cv2
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
def generate_region_masks_5ch(
    landmarks: np.ndarray,
    image_size=(128, 128),
    region_order=('brow', 'eye', 'nose', 'mouth', 'chin'),
    region_landmark_mapping=REGION_LANDMARK_MAPPING_68,
    expand: int = 5
) -> np.ndarray:
    """
    生成 5 通道区域 mask，每个通道对应一个 facial region

    返回：
    - masks: np.ndarray, shape = (5, H, W), dtype uint8
    """
    H, W = image_size
    num_regions = len(region_order)

    masks = np.zeros((num_regions, H, W), dtype=np.uint8)

    for r_idx, region in enumerate(region_order):
        if region not in region_landmark_mapping:
            print(f"[Warning] 未定义区域: {region}")
            continue

        pts = []
        for idx in region_landmark_mapping[region]:
            if idx >= len(landmarks):
                continue
            x, y = landmarks[idx]
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

        masks[r_idx, y1:y2 + 1, x1:x2 + 1] = 1

    return masks
