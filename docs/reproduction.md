# Reproduction

Train per-subject RegionNet checkpoints:

```bash
python scripts/train_regionnet.py \
  --data-root data/casme2/CASME2_LOSO_with_5channel_region \
  --output-dir checkpoints/regionnet
```

Train RA-MoE:

```bash
python scripts/train_ramoe.py --config configs/casme2.yaml
```

The default outputs are written to:

```text
outputs/casme2_ramoe/
```

The implementation keeps the paper-effective branches:

- `S`: global static appearance.
- `T`: optical-flow temporal motion.
- `AU`: RegionNet-guided region branch.

