# CASME II Dataset Preparation

This repository does not redistribute CASME II. Download or request the dataset and annotation file from the official dataset provider, then arrange local files as:

```text
data/casme2/
  CASME2_RAW_selected/
  CASME2-coding-20140508.xlsx
```

Run preprocessing in order:

```bash
python scripts/preprocess/01_crop_faces.py \
  --raw-root data/casme2/CASME2_RAW_selected \
  --output-root data/casme2/CASME2_Cropped128 \
  --retinaface-model pretrained/Resnet50_Final.pth

python scripts/preprocess/02_extract_keyframes.py \
  --annotation-file data/casme2/CASME2-coding-20140508.xlsx \
  --source-dir data/casme2/CASME2_Cropped128 \
  --target-dir data/casme2/CASME2_onset_apex_offset

python scripts/preprocess/03_generate_optical_flow.py \
  --input-folder data/casme2/CASME2_onset_apex_offset \
  --output-folder data/casme2/CASME2_optflow_retinaface

python scripts/preprocess/04_generate_region_labels.py \
  --annotation-file data/casme2/CASME2-coding-20140508.xlsx \
  --keyframe-root data/casme2/CASME2_onset_apex_offset \
  --output-root data/casme2/CASME2_region_gt

python scripts/preprocess/05_generate_region_masks.py \
  --image-root data/casme2/CASME2_onset_apex_offset \
  --output-root data/casme2/CASME2_masks_5area \
  --retinaface-model pretrained/Resnet50_Final.pth \
  --landmark-model pretrained/shape_predictor_68_face_landmarks.dat
```

After merging keyframes, optical flow, region labels, and region masks into the class-wise directory described in the paper, build LOSO splits:

```bash
python scripts/preprocess/06_build_loso_split.py \
  --data-folder data/casme2/casme2_retinaface_5region \
  --output-folder data/casme2/CASME2_LOSO_with_5channel_region
```

