# MV-AFA SzCORE Submission

Multi-View Adaptive Fusion Attention (MV-AFA) seizure detection algorithm, packaged for the [SzCORE benchmark](https://epilepsybenchmarks.com).

**Current version: v1.3.2** (leaderboard name **MV-AFA_Guo_2026**), with the same weights as v1.3.0.
v1.3.1 fixed the input pipeline ([v1.3.1 fix](#v131-channel-mapping-fix)); v1.3.2 changes only the
event post-processing ([v1.3.2 post-processing](#v132-post-processing)).
Multi-dataset model (CHB-MIT + Siena + TUH-Sz),
with TUSZ expanded to 1500 recordings for greater subject diversity. The SzCORE
leaderboard scores event-F1 across 5 datasets, so a model that has seen multiple
corpora beats a CHB-MIT-only specialist on the aggregate (see `Cross-dataset
notes` below).

- 🐳 Docker: `docker.io/mellow99/mv-afa-szcore:v1-3-2` (the tag has no dots so it matches the key the SzCORE website derives from the image name; `v1-3-1` is the input fix alone)
- 🔀 SzCORE PRs: [esl-epfl/szcore#89](https://github.com/esl-epfl/szcore/pull/89) (v1.3.0, merged) · [esl-epfl/szcore#93](https://github.com/esl-epfl/szcore/pull/93) (v1.3.1 fix, under review)
- 🏆 v1.3.0 on the leaderboard: [epilepsybenchmarks.com, MV-AFA v1.3.0](https://epilepsybenchmarks.com/algorithm/?algo=docker-io-mellow99-mv-afa-szcore-v1.3.0) (affected by the bug below; v1.3.1 re-evaluation pending)

## v1.3.1 channel-mapping fix

The v1.3.0 leaderboard numbers below were produced with a broken input stage:

| Input | v1.3.0 | v1.3.1 |
|---|---|---|
| SzCORE referential channels `Fp1-Avg`, `T3-Avg`, … (Siena, TUH, Dianalund, SeizeIT) | `-Avg` suffix not stripped, **18/18 bipolar pairs zero-filled**; the model sees a flat signal and outputs a constant p≈0.95, giving one event over the whole recording (≈288 FP/day) | all 18 pairs built |
| SzCORE CHB-MIT files `chb12_27`, `chb12_28`, `chb12_29` (re-derived from unipolar, so labelled `F7-T3`, `T3-T5`, `T5-O1`, `F8-T4`, `T4-T6`, `T6-O2`) | aliases applied only to whole names, **6/18 temporal pairs zero-filled** | all 18 pairs built |

All other SzCORE CHB-MIT files keep the original CHB-MIT labels (`F7-T7`, ...), which v1.3.0 already handled, so the
v1.3.0 CHB-MIT score is valid except for those three files (checked with `epilepsy2bids` 0.0.7, the converter SzCORE uses).

Fix in `algo/mvafa_szcore/eeg_io.py`: reference suffixes (`AVG`, `REF`, `LE`, `AR`, `CAR`) are stripped and
the T3/T4/T5/T6 → T7/T8/P7/P8 aliases are applied to each electrode of a bipolar name. The loader now
raises an error if no bipolar pair can be built, instead of silently predicting on zeros.

Local check (SzCORE-format EDFs made from the original recordings; outputs of the fixed code are
identical to those on the original EDFs):

| Recording | v1.3.0 on SzCORE format | v1.3.1 on SzCORE format |
|---|---|---|
| Siena PN00-1 | 18/18 zero pairs, p=0.949 for every window, 1 event covering 8–2614 s | P(seizure)=0.81, P(background)=0.65, seizure hit, 31.6 FP/h |
| Siena PN12-3 | (same collapse) | seizure hit, 0 FP |
| CHB-MIT chb11_92 | 0/18 zero pairs (original labels; identical to v1.3.1) | P(seizure)=0.92, P(background)=0.22, seizure hit, 1 FP/h |

Siena PN00 still has a high background probability after the fix, which is a model/calibration
limitation rather than an I/O bug.

## v1.3.2 post-processing

With all 18 pairs built, some recordings still sit above the fixed 0.70 threshold for most of their
length (Siena PN00: background median p≈0.72), and the fixed threshold turns that into an event every
few minutes. v1.3.2 keeps the model and changes only `probs_to_events`, depending on the recording
length:

- **recordings of at least 60 min**: threshold 0.60 and smoothing over 5 windows, with a
  **recording-adaptive gate** that raises the threshold to the 99.5th percentile of that recording's
  smoothed probabilities, so at most 0.5 % of its windows can be positive. An event needs at least 3
  consecutive positive windows, so this caps false alarms at about 36 per day of recording, whatever the
  background level. The gate only ever raises the threshold.
- **shorter recordings** (every TUH-Sz file): the v1.3.1 setting, threshold 0.70 and smoothing over 7
  windows, ungated. They are too short for a background baseline, and without the gate the lower
  threshold doubled TUH-Sz false alarms (33 → 74 per day).
- minimum duration 10 s and merge gap 5 s are unchanged.

The long-recording setting was chosen from 777 candidates (fixed thresholds, smoothing, minimum duration,
and gate quantile / minimum recording length) by mean subject-averaged event F1 over CHB-MIT, Siena and
TUH-Sz. Leave-one-dataset-out (LODO) selection, i.e. choosing on two corpora and scoring the third, picked
the same setting when CHB-MIT or Siena was held out. Keeping the v1.3.1 setting for short recordings was
decided afterwards, from the TUH-Sz result.

Local reproduction: SzCORE-format EDFs made with `epilepsy2bids` 0.0.7 (the converter SzCORE uses),
scored with `timescoring` 426f8d2 event scoring (tolerances 30 s / 60 s, events capped at 5 min, 90 s
between events). Counts are summed per subject and averaged over subjects, like `szcore-evaluation`.
The local v1.3.0 run reproduces the Siena leaderboard result exactly.

| Dataset (local set) | Version | Sens | Prec | F1 | FP/day |
|---|---|---|---|---|---|
| CHB-MIT: all 141 seizure files + 2 seizure-free files per subject (189 files, 23 subjects, 267 h) | v1.3.0 | 0.749 | 0.685 | 0.653 | 10.0 |
| | v1.3.1 | 0.746 | 0.698 | 0.657 | 8.7 |
| | **v1.3.2** | 0.716 | 0.840 | **0.741** | **2.1** |
| Siena: all 41 files (14 subjects, 141 h) | v1.3.0 | 1.000 | 0.037 | 0.071 | 274.7 |
| | v1.3.1 | 0.555 | 0.572 | 0.422 | 17.5 |
| | **v1.3.2** | 0.688 | 0.533 | **0.546** | **16.4** |
| TUH-Sz eval: 864 files (43 subjects, 128 h) | v1.3.0 | 1.000 | 0.182 | 0.270 | 257.5 |
| | v1.3.1 | 0.475 | 0.423 | 0.363 | 33.0 |
| | **v1.3.2** (same as v1.3.1) | 0.475 | 0.423 | 0.363 | 33.0 |
| | rejected: long-recording setting on short recordings too | 0.684 | 0.342 | 0.390 | 73.6 |

v1.3.2 is at least as good as v1.3.1 on every corpus, in both F1 and FP/day.

Read these with care:

- CHB-MIT, Siena and TUH-Sz are **training corpora** of the model, and the long-recording setting was
  chosen on them too. LODO selection picked the same setting for the CHB-MIT and Siena folds. On TUH-Sz,
  v1.3.2 runs the v1.3.1 setting, which was not tuned on SzCORE data.
- The local CHB-MIT set has only 48 of the seizure-free files, so its precision and FP/day are better than
  on the full set (leaderboard v1.3.0: F1 0.520, 12.8 FP/day). On those 48 files (77 h) false alarms drop
  from 8.4/day (v1.3.1) to 4.3/day (v1.3.2).
- No TUH-Sz file is 60 min long, so on TUH-Sz v1.3.2 gives exactly the v1.3.1 output.
- Siena recordings shorter than 60 min (6 of 41) also keep the v1.3.1 setting; PN00-1 (44 min) gets its
  seizure and 9 false alarms (25 predicted events, merged by the scorer).
- The held-out corpora (Dianalund, SeizeIT1) cannot be evaluated locally. Their recordings are long, so the
  gate applies to them.

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
FP/day). They say nothing about cross-dataset generalization. The CHB-MIT score is valid apart from
chb12_27-29, which ran with 6 of 18 pairs zeroed. Generalization claims should be based only on re-evaluated (v1.3.1 or later) results for the
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

## Tests

`tests/test_eeg_io.py` writes small EDFs in the SzCORE formats (19 `<elec>-Avg` common-average channels,
and 18 double-banana bipolar channels with old or new 10-20 names) and checks that all 18 bipolar pairs are
built with the right values, that an unrecognised montage raises, that the model output on SzCORE unipolar
input is not constant, and that smoothing keeps one value per window. v1.3.0 fails 10 of these tests.
`tests/test_postprocessing.py` checks the v1.3.2 post-processing: the gate removes the false alarms of a long
recording whose background is above the threshold and only ever removes events, and recordings under 60 min
(or of unknown length) get exactly the v1.3.1 setting.

```bash
pip install pytest
python -m pytest tests
```

## Build & test the Docker image

> ⚠️ **Build for `linux/amd64`.** SzCORE CI runs on linux/amd64. On Apple Silicon
> a plain `docker build` produces an **arm64** image that the benchmark cannot
> run ("cannot pull the image"). Always cross-build and push with buildx:
> ```bash
> docker buildx build --platform linux/amd64 -t mellow99/mv-afa-szcore:v1-3-2 --push .
> docker manifest inspect mellow99/mv-afa-szcore:v1-3-2   # must show linux/amd64
> ```
> Also make sure the Docker Hub repo is **public** (it defaults to private).

```bash
# On linux/amd64 hosts a plain build is fine:
docker build -t mellow99/mv-afa-szcore:v1-3-2 .

# Test on any EDF file:
docker run --rm \
  -v /path/to/edf_dir:/data \
  -v /tmp/szcore_out:/output \
  -e INPUT=sample.edf \
  -e OUTPUT=sample.tsv \
  mellow99/mv-afa-szcore:v1-3-2
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
docker push mellow99/mv-afa-szcore:v1-3-2

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
window-level generalization. Event-F1 operating point up to v1.3.1: threshold 0.70,
smoothing 7 (v1.3.2: see [v1.3.2 post-processing](#v132-post-processing)). (Both multi-dataset models vastly outperform the CHB-MIT-only
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
- **Post-processing**: on recordings of 60 min or more, per-window probabilities are smoothed over 5 windows (persistence filter) and thresholded at 0.60, or at the recording's 99.5th percentile if higher; shorter recordings are smoothed over 7 windows and thresholded at 0.70. Events are then filtered by minimum duration 10 s and merge gap 5 s.
- **Weights**: multi-dataset, trained on CHB-MIT + Siena + TUH-Sz (TUSZ 1500 recordings; subject-level splits, EEG augmentation, per-dataset balancing).
