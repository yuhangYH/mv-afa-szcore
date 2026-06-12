# MV-AFA SzCORE Submission

Multi-View Adaptive Fusion Attention (MV-AFA) seizure detection algorithm, packaged for the [SzCORE benchmark](https://epilepsybenchmarks.com).

## Package structure

```
mv_afa_szcore/
├── Dockerfile
├── mv_afa.yaml              # SzCORE submission descriptor (edit before submitting)
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
        ├── inference.py     # Sliding-window inference + event merging
        └── weights/
            └── best_model.pt
```

## Step 1 — (Optional) Train a cross-subject model

The bundled `weights/best_model.pt` is trained on **chb01 only** (single subject).
For proper cross-dataset performance on SzCORE, train on all subjects first:

```bash
# inside the MV-AFA training directory
python eeg_chbmit_multiview_gated.py  # edit subject_id to cover all subjects

# then replace weights:
python scripts/prepare_weights.py \
    --weights-src /path/to/cross_subject_training/best_model.pt
```

## Step 2 — Build the Docker image

```bash
cd /Users/yuhang/Downloads/mv_afa_szcore
docker build -t YOUR_DOCKERHUB_USERNAME/mv-afa-szcore:v1.0.0 .
```

Test locally with any EDF file:

```bash
docker run --rm \
  -v /path/to/edf_dir:/data \
  -v /tmp/szcore_out:/output \
  -e INPUT=sample.edf \
  -e OUTPUT=sample.tsv \
  YOUR_DOCKERHUB_USERNAME/mv-afa-szcore:v1.0.0
cat /tmp/szcore_out/sample.tsv
```

Expected output TSV (BIDS format):
```
onset   duration   eventType   confidence   channels   dateTime   recordingDuration
123.0   45.0       sz          n/a          n/a        2010-...   3600.0
```

## Step 3 — Push to Docker Hub

```bash
docker login
docker push YOUR_DOCKERHUB_USERNAME/mv-afa-szcore:v1.0.0
```

## Step 4 — Update mv_afa.yaml

Edit `mv_afa.yaml` and replace:
- `YOUR_DOCKERHUB_USERNAME` → your actual Docker Hub username
- `YOUR_GITHUB_USERNAME` → your actual GitHub username
- `YOUR INSTITUTION` → your affiliation

## Step 5 — Submit to SzCORE

```bash
# Fork the SzCORE repo on GitHub, then:
git clone https://github.com/YOUR_GITHUB_USERNAME/szcore
cp mv_afa.yaml szcore/algorithms/mv_afa.yaml
cd szcore
git checkout -b add-mv-afa
git add algorithms/mv_afa.yaml
git commit -m "Add MV-AFA seizure detection algorithm"
git push origin add-mv-afa
# Open a Pull Request → SzCORE CI runs Docker image automatically
```

## Notes

- **Channel remontage**: SzCORE standardizes to 19 standard 10-20 channels. The inference code automatically derives the 18 CHB-MIT bipolar pairs algebraically.
- **Inference parameters**: 2 s windows, 1 s step, threshold=0.54, min seizure duration=5 s, merge gap=30 s.
- **Known limitation**: Current weights are single-subject (chb01). Cross-subject generalization will be limited until a cross-subject model is trained.
