import argparse
import glob
import os
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from third_party.RetinaFace.tools import FaceDetector


def crop_images_casme2_retinaface(raw_root, output_root, retinaface_model):
    face_detection = FaceDetector(retinaface_model)

    for subject_name in sorted(os.listdir(raw_root)):
        subject_path = os.path.join(raw_root, subject_name)
        output_subject_path = os.path.join(output_root, subject_name)

        if not os.path.isdir(subject_path):
            continue

        print(f"Processing subject: {subject_name}")
        for sequence_name in sorted(os.listdir(subject_path)):
            sequence_path = os.path.join(subject_path, sequence_name)
            output_sequence_path = os.path.join(output_subject_path, sequence_name)

            if not os.path.isdir(sequence_path):
                continue

            print(f"  Sequence: {sequence_name}")
            os.makedirs(output_sequence_path, exist_ok=True)

            face_left = face_top = face_right = face_bottom = None
            for index, img_file_path in enumerate(sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))):
                image = cv2.imread(img_file_path)
                if image is None:
                    print(f"Warning: failed to read {img_file_path}")
                    continue

                if index == 0:
                    coords = face_detection.cal(image)
                    if coords is None or len(coords) != 4:
                        print(f"Warning: no face detected in {img_file_path}")
                        break
                    face_left, face_top, face_right, face_bottom = coords

                face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]
                face = cv2.resize(face, (128, 128))
                output_img_path = os.path.join(output_sequence_path, os.path.basename(img_file_path))
                cv2.imwrite(output_img_path, face)

    print("Face cropping and saving complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop CASME II faces to 128x128 with RetinaFace.")
    parser.add_argument("--raw-root", required=True, help="CASME II raw selected sequence root")
    parser.add_argument("--output-root", required=True, help="output directory for cropped faces")
    parser.add_argument("--retinaface-model", required=True, help="RetinaFace Resnet50_Final.pth path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crop_images_casme2_retinaface(args.raw_root, args.output_root, args.retinaface_model)
