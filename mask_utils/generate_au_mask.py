import numpy as np
import cv2

# 基于 68 点的 AU 映射表
AU_LANDMARK_MAPPING_68 = {
    'AU01': [19, 20, 21, 22],                  # inner brow raiser，眉头内侧区域
    'AU02': [17, 18, 21, 22, 23, 24],          # outer brow raiser，眉头外侧区域
    'AU04': list(range(17, 27)) + [27],        # brow lowerer，整个眉区
    'AU05': [37, 38, 43, 44],                  # upper lid raiser，眼睛上缘
    'AU06': [36, 39, 42, 45],                  # cheek raiser，眼部周围
    'AU07': list(range(37, 42)) + list(range(42, 48)),  # lid tightener，眼睛周边闭合区域
    'AU09': list(range(29, 36)),               # nose wrinkle，鼻翼周围
    'AU10': list(range(48, 60)),               # upper lip raiser，整个嘴唇区域
    'AU12': list(range(48, 60)),               # lip corner puller，整个嘴唇区域
    'AU14': list(range(48, 60)),               # dimpler，整个嘴唇区域
    'AU15': list(range(48, 60)),               # lip corner depressor，整个嘴唇区域
    'AU16': list(range(48, 60)),               # lower lip depressor，整个嘴唇区域
    'AU17': [7, 8, 9, 10, 11, 12, 56, 57, 58, 59, 60],  # chin raiser，下巴及下唇边缘
    'AU18': list(range(48, 60)),               # lip pucker，整个嘴唇区域
    'AU20': list(range(48, 60)),               # lip stretcher，整个嘴唇区域
    'AU23': list(range(48, 60)),               # lip pressor，整个嘴唇区域
    'AU24': list(range(48, 60)),               # lip pressor，整个嘴唇区域
    'AU25': list(range(48, 60)),               # lips part，整个嘴唇区域
    'AU26': [7, 8, 9, 10, 11, 12, 51, 52, 53, 56, 57, 58, 59, 60],  # jaw drop，下巴及上下唇边缘
    'AU38': [29, 30, 31, 32, 33, 34]          # nostril dilator，鼻孔周围
}


def generate_au_mask(
    landmarks: np.ndarray,
    active_aus: list,
    image_size=(224, 224),
    au_landmark_mapping=AU_LANDMARK_MAPPING_68,
    expand: int = 5
) -> np.ndarray:
    """
    根据活动的 AU 列表，使用对应的 landmarks，生成 mask。

    参数：
    - landmarks: np.ndarray, [68, 2] 关键点坐标
    - active_aus: list, 如 ['AU04', 'AU12']
    - image_size: tuple, (高, 宽)
    - au_landmark_mapping: dict, AU 对应的关键点索引映射
    - expand: int, 扩展边界的像素数

    返回：
    - mask: np.ndarray, 二值图，关键区域为 1，其余为 0
    """
    H, W = image_size
    mask = np.zeros((H, W), np.uint8)

    for au in active_aus:
        if au not in au_landmark_mapping:
            print(f"Warning: 未定义的 AU {au}")
            continue

        pts = []
        for idx in au_landmark_mapping[au]:
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

        mask[y1:y2 + 1, x1:x2 + 1] = 1

    return mask

if __name__ == "__main__":
    print("generate_au_mask 模块仅用于导入使用，不应直接执行。")
