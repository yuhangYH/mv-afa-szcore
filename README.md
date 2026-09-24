# MV-AFA SzCORE Submission

Multi-View Adaptive Fusion Attention (MV-AFA) seizure detection algorithm, packaged for the [SzCORE benchmark](https://epilepsybenchmarks.com).

**Current release: v1.3.1** (leaderboard name **MV-AFA_Guo_2026**). This is an input-pipeline bug fix with
the same v1.3.0 weights; see [v1.3.1 fix](#v131-channel-mapping-fix) below.
Multi-dataset model (CHB-MIT + Siena + TUH-Sz),
with TUSZ expanded to 1500 recordings for greater subject diversity. The SzCORE
leaderboard scores event-F1 across 5 datasets, so a model that has seen multiple
corpora beats a CHB-MIT-only specialist on the aggregate (see `Cross-dataset
notes` below).

- 🐳 Docker: `docker.io/mellow99/mv-afa-szcore:v1-3-1` (the tag has no dots so it matches the key the SzCORE website derives from the image name)
- 🔀 SzCORE PR: [esl-epfl/szcore#89](https://github.com/esl-epfl/szcore/pull/89) — **merged ✅**
- 🏆 v1.3.0 on the leaderboard: [epilepsybenchmarks.com, MV-AFA v1.3.0](https://epilepsybenchmarks.com/algorithm/?algo=docker-io-mellow99-mv-afa-szcore-v1.3.0) (affected by the bug below; v1.3.1 re-evaluation pending)

## v1.3.1 channel-mapping fix

The v1.3.0 leaderboard numbers below were produced with a broken input stage:

| Input | v1.3.0 | v1.3.1 |
|---|---|---|
| SzCORE referential channels `Fp1-Avg`, `T3-Avg`, … (Siena, TUH, Dianalund, SeizeIT) | `-Avg` suffix not stripped, **18/18 bipolar pairs zero-filled**; the model sees a flat signal and outputs a constant p≈0.95, giving one event over the whole recording (≈288 FP/day) | all 18 pairs built |
| SzCORE CHB-MIT bipolar channels with old labels `F7-T3`, `T3-T5`, `T5-O1`, `F8-T4`, `T4-T6`, `T6-O2` | aliases applied only to whole names, **6/18 temporal pairs zero-filled** | all 18 pairs built |

Fix in `algo/mvafa_szcore/eeg_io.py`: reference suffixes (`AVG`, `REF`, `LE`, `AR`, `CAR`) are stripped and
the T3/T4/T5/T6 → T7/T8/P7/P8 aliases are applied to each electrode of a bipolar name. The loader now
raises an error if no bipolar pair can be built, instead of silently predicting on zeros.

Local check (SzCORE-format EDFs made from the original recordings; outputs of the fixed code are
identical to those on the original EDFs):

| Recording | v1.3.0 on SzCORE format | v1.3.1 on SzCORE format |
|---|---|---|
| Siena PN00-1 | 18/18 zero pairs, p=0.949 for every window, 1 event covering 8–2614 s | P(seizure)=0.81, P(background)=0.65, seizure hit, 31.6 FP/h |
| Siena PN12-3 | (same collapse) | seizure hit, 0 FP |
| CHB-MIT chb11_92 | 6/18 zero pairs | P(seizure)=0.92, P(background)=0.22, seizure hit, 1 FP/h |

Siena PN00 still has a high background probability after the fix, which is a model/calibration
limitation rather than an I/O bug.

## 🏆 Official SzCORE leaderboard results for v1.3.0 (event-based F1)

The submission is merged and fully evaluated across all five benchmark datasets.
🚂 marks a **training** dataset (not counted as generalization); Dianalund and
SeizeIT are the true held-out corpora.

| Dataset | F1 (%) | Sensitivity (%) | Precision (%) | FP / day |
|---------|:------:|:---------------:|:-------------:|:--------:|
| CHB-MIT 🚂 | **52.00** | 74.92 | 46.25 | **13** |
| TUH 🚂 | 17.68 | 98.16 | 13.13 | 283 |
| Siena 🚂 | 7.05 | 100.00 | 3.74 | 275 |
| Dianalund | 2.18 | 100.00 | 1.28 | 290 |
| SeizeIT | 0.77 | 100.00 | 0.39 | 288 |

**These v1.3.0 numbers are invalid outside CHB-MIT.** The ≈100 % sensitivity and ≈288 FP/day on
Siena, TUH, Dianalund and SeizeIT are the signature of a constant "seizure" output caused by the
zero-filled input (the scorer caps events at 5 min, so an always-on detector gives 86400/300 ≈ 288
FP/day). They say nothing about cross-dataset generalization. Even the CHB-MIT score ran with 6 of
18 channels zeroed. Generalization claims should be based only on the v1.3.1 results for the
held-out corpora **Dianalund and SeizeIT1**; CHB-MIT, Siena and TUH are training corpora.

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
            └── best_model.pt   # multi-dataset (CHB-MIT + Siena + TUSZ)
```

## Build & test the Docker image

> ⚠️ **Build for `linux/amd64`.** SzCORE CI runs on linux/amd64. On Apple Silicon
> a plain `docker build` produces an **arm64** image that the benchmark cannot
> run ("cannot pull the image"). Always cross-build and push with buildx:
> ```bash
> docker buildx build --platform linux/amd64 -t mellow99/mv-afa-szcore:v1-3-1 --push .
> docker manifest inspect mellow99/mv-afa-szcore:v1-3-1   # must show linux/amd64
> ```
> Also make sure the Docker Hub repo is **public** (it defaults to private).

```bash
# On linux/amd64 hosts a plain build is fine:
docker build -t mellow99/mv-afa-szcore:v1-3-1 .

# Test on any EDF file:
docker run --rm \
  -v /path/to/edf_dir:/data \
  -v /tmp/szcore_out:/output \
  -e INPUT=sample.edf \
  -e OUTPUT=sample.tsv \
  mellow99/mv-afa-szcore:v1-3-1
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
docker push mellow99/mv-afa-szcore:v1-3-1

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

## Validated performance (window-level AUROC on held-out subjects)

v1.3.0 (TUSZ 1500) vs v1.2.0 (TUSZ 400), held-out-subject window AUROC:

| Dataset | v1.2.0 | v1.3.0 |
|---------|--------|--------|
| CHB-MIT | 0.794 | 0.820 |
| Siena | 0.755 | 0.762 |
| TUH-Sz (TUSZ) | 0.658 | 0.675 |
| overall | 0.732 | **0.745** |

Expanding TUSZ training data (400→1500 recordings) consistently improved
window-level generalization. Event-F1 operating point: threshold 0.70,
smoothing 7. (Both multi-dataset models vastly outperform the CHB-MIT-only
v1.1.0, which scored ~0.00 on Siena/TUSZ.)

## Cross-dataset notes

- SzCORE evaluates 5 datasets (CHB-MIT, Siena, TUH-Sz, Dianalund, SeizeIT1) by
  event-F1, all standardized to the 19-channel 10-20 montage. Leave-one-dataset-out
  experiments confirmed that **zero-shot transfer to an unseen corpus is near
  chance** — a model must be *trained* on each target corpus. This model trains on
  CHB-MIT + Siena + TUSZ (3 of 5).
- **Dianalund** is a private held-out benchmark (not downloadable). **SeizeIT1**
  is standardized to scalp 10-20 by SzCORE (so this model is architecturally
  compatible) but its access is closed (ethical approval expired), so it cannot be
  added to training. Both are therefore evaluated zero-shot (~chance) by this model.

## Notes

- **Channel remontage**: SzCORE standardizes to 19 standard 10-20 channels named `<electrode>-Avg`; the inference code strips the reference suffix and derives the 18 CHB-MIT bipolar pairs algebraically. It also falls back to bipolar channel names (e.g. `FP1-F7`, or SzCORE CHB-MIT's `F7-T3`) when given an already-bipolar recording.
- **Signal scaling**: signals are read in MNE Volts with **no extra rescaling and no filtering**, matching the training pipeline exactly (per-window z-scoring is applied inside feature extraction).
- **Inference parameters**: 2 s windows, 4 s step, TDA folds = 5.
- **Post-processing**: per-window probabilities are smoothed over 7 windows (persistence filter), thresholded at 0.70, then filtered by minimum duration 10 s and merge gap 5 s.
- **Weights**: multi-dataset, trained on CHB-MIT + Siena + TUH-Sz (TUSZ 1500 recordings; subject-level splits, EEG augmentation, per-dataset balancing).
