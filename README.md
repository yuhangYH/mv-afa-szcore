# MV-AFA SzCORE Submission

Multi-View Adaptive Fusion Attention (MV-AFA) seizure detection algorithm, packaged for the [SzCORE benchmark](https://epilepsybenchmarks.com).

**Current release: v1.2.0** — multi-dataset model (CHB-MIT + Siena + TUH-Sz),
chosen because the SzCORE leaderboard scores event-F1 across 5 datasets, so a
model that has seen multiple corpora beats a CHB-MIT-only specialist on the
aggregate (see `Cross-dataset notes` below).

- 🐳 Docker: `docker.io/mellow99/mv-afa-szcore:v1.2.0`
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

## Validated performance (v1.2.0, event-level F1 on held-out subjects)

Operating point: threshold 0.70, smoothing window 7. Event-based (overlap) F1
on continuous recordings of the held-out validation subjects:

| Dataset | Event F1 |
|---------|----------|
| CHB-MIT | 0.29 |
| Siena | 0.24 |
| TUH-Sz (TUSZ) | 0.50 |
| 3-dataset overall | **0.41** |

For comparison, the CHB-MIT-only v1.1.0 scored F1 0.68 on CHB-MIT but ~0.00 on
Siena/TUSZ (overall 0.26) — the multi-dataset model is clearly better on the
SzCORE-style multi-dataset aggregate.

## Cross-dataset notes

- SzCORE evaluates 5 datasets (CHB-MIT, Siena, TUH-Sz, Dianalund, SeizeIT1) by
  event-F1. Leave-one-dataset-out experiments confirmed that **zero-shot transfer
  to an unseen corpus is near chance** — a model must be *trained* on each target
  corpus. v1.2.0 therefore trains on CHB-MIT + Siena + TUSZ (3 of 5).
- SeizeIT1 (wearable behind-the-ear, 2 channels) is out of scope for this
  18-channel 10-20 model; Dianalund was not available locally.

## Notes

- **Channel remontage**: SzCORE standardizes to 19 standard 10-20 channels; the inference code derives the 18 CHB-MIT bipolar pairs algebraically. It also falls back to bipolar channel names (e.g. `FP1-F7`) when given an already-bipolar recording.
- **Signal scaling**: signals are read in MNE Volts with **no extra rescaling and no filtering**, matching the training pipeline exactly (per-window z-scoring is applied inside feature extraction).
- **Inference parameters**: 2 s windows, 4 s step, TDA folds = 5.
- **Post-processing**: per-window probabilities are smoothed over 7 windows (persistence filter), thresholded at 0.70, then filtered by minimum duration 10 s and merge gap 5 s.
- **Weights**: multi-dataset, trained on CHB-MIT + Siena + TUH-Sz (subject-level splits, EEG augmentation, per-dataset balancing).
