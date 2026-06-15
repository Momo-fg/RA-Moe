import argparse
import os

import numpy as np
import pandas as pd


AU_TO_REGION = {
    "AU01": "brow",
    "AU02": "brow",
    "AU04": "brow",
    "AU05": "eye",
    "AU06": "eye",
    "AU07": "eye",
    "AU09": "nose",
    "AU38": "nose",
    "AU10": "mouth",
    "AU12": "mouth",
    "AU14": "mouth",
    "AU15": "mouth",
    "AU16": "mouth",
    "AU18": "mouth",
    "AU20": "mouth",
    "AU23": "mouth",
    "AU24": "mouth",
    "AU25": "mouth",
    "AU17": "chin",
    "AU26": "chin",
}
REGION_TO_INDEX = {
    "brow": 0,
    "eye": 1,
    "nose": 2,
    "mouth": 3,
    "chin": 4,
}
NUM_REGIONS = 5


def read_annotations(file_path):
    df = pd.read_excel(file_path, engine="openpyxl", header=0, dtype=str)
    return df.fillna("")


def parse_aus(aus_str):
    aus_list = []
    unparsed = []
    for au in aus_str.split("+"):
        au = au.strip().upper().replace("L", "").replace("R", "")
        if au.isdigit():
            aus_list.append(f"AU{int(au):02d}")
        elif au and au != "?":
            unparsed.append(au)
    return aus_list, unparsed


def generate_region_labels(annotation_file, keyframe_root, output_root):
    annotations = read_annotations(annotation_file)
    os.makedirs(output_root, exist_ok=True)

    unparsed_aus = set()
    missing_images = set()

    for idx, row in annotations.iterrows():
        print(f"Processing row {idx + 1}/{len(annotations)}")

        subject_id = row.get("Subject", "").strip()
        filename = row.get("Filename", "").strip()
        emotion = row.get("Estimated Emotion", "").strip()
        aus_str = row.get("Action Units", "").strip()

        if not subject_id or not filename or not emotion:
            print("Warning: missing required annotation fields; skipped")
            continue

        try:
            subject = f"sub{int(subject_id):02d}"
        except ValueError:
            print(f"Invalid subject ID: {subject_id}")
            continue

        aus_list, unparsed = parse_aus(aus_str)
        if unparsed:
            unparsed_aus.add(f"{subject}/{filename}: {'+'.join(unparsed)}")

        for frame in ["onset", "apex", "offset"]:
            img_name = f"{subject}_{filename}_{frame}.jpg"
            img_path = os.path.join(keyframe_root, emotion, img_name)
            if not os.path.exists(img_path):
                missing_images.add(f"{subject}/{filename}/{frame}")

            region_gt = np.zeros(NUM_REGIONS, dtype=np.float32)
            for au in aus_list:
                region = AU_TO_REGION.get(au)
                if region is not None:
                    region_gt[REGION_TO_INDEX[region]] = 1.0

            gt_dir = os.path.join(output_root, emotion)
            os.makedirs(gt_dir, exist_ok=True)
            np.save(os.path.join(gt_dir, f"{subject}_{filename}_{frame}_region.npy"), region_gt)

    print("\n==================== Summary ====================")
    print(f"Annotation rows: {len(annotations)}")
    print(f"Generated region labels: {len(annotations) * 3}")
    if unparsed_aus:
        print(f"Unparsed AU records: {len(unparsed_aus)}")
    if missing_images:
        print(f"Missing keyframe images: {len(missing_images)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5-region supervision labels from CASME II AU annotations.")
    parser.add_argument("--annotation-file", required=True, help="CASME II annotation xlsx")
    parser.add_argument("--keyframe-root", required=True, help="onset/apex/offset keyframe root")
    parser.add_argument("--output-root", required=True, help="output directory for *_region.npy files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_region_labels(args.annotation_file, args.keyframe_root, args.output_root)
