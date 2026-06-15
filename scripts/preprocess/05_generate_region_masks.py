import argparse
import os
from pathlib import Path
import sys

import cv2
import dlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from mask_utils.generate_region_mask5 import generate_region_masks_5ch
from third_party.RetinaFace.tools import FaceDetector


REGION_ORDER = ("brow", "eye", "nose", "mouth", "chin")
VALID_EXT = {".jpg", ".png", ".jpeg", ".bmp"}


def generate_region_masks(image_root, output_root, retinaface_model, landmark_model, image_size=(128, 128)):
    os.makedirs(output_root, exist_ok=True)
    face_detection = FaceDetector(retinaface_model)
    predictor = dlib.shape_predictor(landmark_model)

    processed = 0
    no_face_count = 0
    missing_landmark = 0

    for root, _, files in os.walk(image_root):
        for fname in files:
            if os.path.splitext(fname)[-1].lower() not in VALID_EXT:
                continue

            img_path = os.path.join(root, fname)
            rel_path = os.path.relpath(img_path, image_root)
            save_name = os.path.splitext(rel_path)[0] + "_region5ch.npy"
            save_path = os.path.join(output_root, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: failed to read {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

            coords = face_detection.cal(image)
            if coords is None or len(coords) != 4:
                no_face_count += 1
                continue

            left, top, right, bottom = coords
            rectangle = dlib.rectangle(int(left), int(top), int(right), int(bottom))
            shape = predictor(image_rgb, rectangle)
            if shape.num_parts != 68:
                missing_landmark += 1
                continue

            landmarks = np.array([[pt.x, pt.y] for pt in shape.parts()], dtype=np.float32)
            height, width = image.shape[:2]
            landmarks[:, 0] *= image_size[1] / width
            landmarks[:, 1] *= image_size[0] / height

            mask_5ch = generate_region_masks_5ch(
                landmarks=landmarks,
                image_size=image_size,
                region_order=REGION_ORDER,
            ).astype(np.uint8)
            assert mask_5ch.shape == (5, image_size[0], image_size[1]), f"mask shape error: {mask_5ch.shape}"

            np.save(save_path, mask_5ch)
            processed += 1
            if processed % 200 == 0:
                print(f"Processed {processed} images")

    print("\n==================== Summary ====================")
    print(f"Processed images: {processed}")
    print(f"No-face images: {no_face_count}")
    print(f"Landmark failures: {missing_landmark}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5-channel region masks for CASME II keyframes.")
    parser.add_argument("--image-root", required=True, help="keyframe image root")
    parser.add_argument("--output-root", required=True, help="output directory for *_region5ch.npy files")
    parser.add_argument("--retinaface-model", required=True, help="RetinaFace Resnet50_Final.pth path")
    parser.add_argument("--landmark-model", required=True, help="dlib shape_predictor_68_face_landmarks.dat path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_region_masks(args.image_root, args.output_root, args.retinaface_model, args.landmark_model)
