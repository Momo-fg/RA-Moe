# Pretrained assets

Place external model assets in this directory before running the full pipeline:

- `SKD-TSTSAN.pth`: macro-expression pretrained backbone used by RA-MoE.
- `Resnet50_Final.pth`: RetinaFace face detector checkpoint.
- `magnet.pth`: motion magnification checkpoint.
- `shape_predictor_68_face_landmarks.dat`: dlib 68-point landmark model.

Large third-party weights are intentionally not committed to Git. If release-ready weights are available, publish them through a GitHub Release, Hugging Face, or another stable download location and update this file.

