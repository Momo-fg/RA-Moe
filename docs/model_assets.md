# Model Assets

The code expects external checkpoints under `pretrained/` by default.

```text
pretrained/
  SKD-TSTSAN.pth
  Resnet50_Final.pth
  magnet.pth
  shape_predictor_68_face_landmarks.dat
```

These files are not stored in the repository because they are either large or third-party assets with separate distribution terms. Update `configs/casme2.yaml` if you place them elsewhere.

