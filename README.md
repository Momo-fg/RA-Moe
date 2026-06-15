# RA-MoE: Region-Aware Mixture-of-Experts for AU-Guided Micro-expression Recognition

Official PyTorch implementation of **RA-MoE: Region-Aware Mixture-of-Experts for AU-Guided Micro-expression Recognition**, prepared for the ECML-PKDD 2026 paper release.

![Model Architecture](Architecture.png)

RA-MoE uses three effective branches:

- `S`: global static appearance.
- `T`: optical-flow temporal motion.
- `AU`: RegionNet-guided regional cues.

## Repository Structure

```text
RA-MoE/
  configs/                  # experiment configuration
  docs/                     # dataset, asset, and reproduction notes
  scripts/
    preprocess/             # CASME II preprocessing pipeline
    train_regionnet.py      # RegionNet pretraining
    train_ramoe.py          # RA-MoE training entry
  src/ramoe/
    datasets/               # dataset loaders
    models/                 # RA-MoE and RegionNet models
    training/               # training loop
  third_party/              # bundled third-party source code
  checkpoints/              # generated RegionNet checkpoints
  pretrained/               # external pretrained assets
```

## Installation

```bash
conda env create -f environment.yml
conda activate ramoe
```

or:

```bash
pip install -r requirements.txt
```

`opencv-contrib-python` is required because the optical-flow preprocessing uses TV-L1 optical flow.

## Dataset

CASME II is not redistributed in this repository. Prepare the dataset and annotation file locally:

```text
data/casme2/
  CASME2_RAW_selected/
  CASME2-coding-20140508.xlsx
```

See [docs/dataset_preparation.md](docs/dataset_preparation.md) for the full preprocessing pipeline.

## Pretrained Assets

Place required external assets under `pretrained/`:

```text
pretrained/
  SKD-TSTSAN.pth
  Resnet50_Final.pth
  magnet.pth
  shape_predictor_68_face_landmarks.dat
```

See [docs/model_assets.md](docs/model_assets.md) for details.

## Preprocessing

Run the CASME II pipeline in order:

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

After merging the generated keyframes, optical flow, region labels, and masks into a class-wise directory, build LOSO splits:

```bash
python scripts/preprocess/06_build_loso_split.py \
  --data-folder data/casme2/casme2_retinaface_5region \
  --output-folder data/casme2/CASME2_LOSO_with_5channel_region
```

## Training

Train RegionNet:

```bash
python scripts/train_regionnet.py \
  --data-root data/casme2/CASME2_LOSO_with_5channel_region \
  --output-dir checkpoints/regionnet
```

Train RA-MoE:

```bash
python scripts/train_ramoe.py --config configs/casme2.yaml
```

Outputs are saved to `outputs/casme2_ramoe/` by default.

## Citation

```bibtex
@inproceedings{
}
```

## Acknowledgements

This repository includes adapted code for RetinaFace-based face detection and motion magnification. Please follow the licenses and citation requirements of the corresponding upstream projects.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
