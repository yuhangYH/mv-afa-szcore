# MV-AFA SzCORE Submission

Multi-View Adaptive Fusion Attention (MV-AFA) seizure detection algorithm, packaged for the [SzCORE benchmark](https://epilepsybenchmarks.com).

**Current release: v1.1.0** — cross-subject model trained on all 24 CHB-MIT subjects.

- 🐳 Docker: `docker.io/mellow99/mv-afa-szcore:v1.1.0`
- 🔀 SzCORE PR: [esl-epfl/szcore#89](https://github.com/esl-epfl/szcore/pull/89)

## Method

A 2-second EEG window is encoded by four complementary branches and fused by a gated mixture (~1.75 M parameters):

1. **Temporal** — multi-scale patch Transformer (patch sizes 8/16/32)
2. **Frequency** — multi-scale 2-D CNN on Welch PSD maps
3. **Statistical** — 40-D hand-crafted features (Hjorth, entropy, line-length, ZCR, …)
4. **Topological (TDA)** — 12-D features from Vietoris–Rips persistent homology (H0 + H1)

## Package structure

```
mv_afa_szcore/
├── Dockerfile
├── mv_afa.yaml              # SzCORE submission descriptor
├── README.md
├── scripts/
│   └── prepare_weights.py   # Helper to copy new weights into the package
└── algo/
    ├── requirements.txt
    └── mvafa_szcore/
        ├── __init__.py
        ├── __main__.py      # SzCORE entrypoint
        ├── model.py         # MultiViewSeizureNet architecture
        ├── features.py      # Statistical / TDA / frequency extraction
        ├── eeg_io.py        # EDF loading + 19→18 bipolar remontage
        ├── inference.py     # Sliding-window inference + smoothed event detection
        └── weights/
            └── best_model.pt   # cross-subject (all 24 CHB-MIT subjects)
```

## Build & test the Docker image

```bash
docker build -t mellow99/mv-afa-szcore:v1.1.0 .

# Test on any EDF file:
docker run --rm \
  -v /path/to/edf_dir:/data \
  -v /tmp/szcore_out:/output \
  -e INPUT=sample.edf \
  -e OUTPUT=sample.tsv \
  mellow99/mv-afa-szcore:v1.1.0
cat /tmp/szcore_out/sample.tsv
```

Expected output TSV (BIDS format):
```
onset   duration   eventType   confidence   channels   dateTime   recordingDuration
444.0   74.0       sz          n/a          n/a        2059-...   3600.0
```

## Push & submit

```bash
docker login
docker push mellow99/mv-afa-szcore:v1.1.0

# Submit: fork esl-epfl/szcore, copy mv_afa.yaml to algorithms/, open a PR.
# SzCORE CI then runs the Docker image automatically.
```

## Retraining (cross-subject)

The bundled weights are already trained cross-subject on all 24 CHB-MIT subjects.
To retrain (e.g. on more data) and refresh the weights:

```bash
# Train (window 2 s / step 4 s, TDA folds 5, neg:pos 4:1):
MVAFA_NEG_POS_RATIO=4.0 python train_cross_subject.py \
    --subject_id all --data_dir <chbmit_root> \
    --output_dir ./run --epochs 30 \
    --balance_method undersample --precompute_features --tda_folds 5

# Swap weights into the package, then rebuild the image:
python scripts/prepare_weights.py --weights-src ./run/best_model.pt
```

## Validated performance

On ~63 h of continuous CHB-MIT recordings (5 subjects), deployed configuration:

| Metric | Value |
|--------|-------|
| Cross-subject AUROC (window-level) | 0.936 |
| Event sensitivity | 21/25 = 84% |
| False alarms | ~0.8 / 24 h |

## Notes

- **Channel remontage**: SzCORE standardizes to 19 standard 10-20 channels; the inference code derives the 18 CHB-MIT bipolar pairs algebraically. It also falls back to bipolar channel names (e.g. `FP1-F7`) when given an already-bipolar recording.
- **Signal scaling**: signals are read in MNE Volts with **no extra rescaling and no filtering**, matching the training pipeline exactly (per-window z-scoring is applied inside feature extraction).
- **Inference parameters**: 2 s windows, 4 s step, TDA folds = 5.
- **Post-processing**: per-window probabilities are smoothed over 7 windows (persistence filter), thresholded at 0.95, then filtered by minimum duration 10 s and merge gap 5 s. This brings false alarms from ~500/24 h down to < 1/24 h while keeping high event sensitivity.
- **Weights**: cross-subject, trained on all 24 CHB-MIT subjects.
