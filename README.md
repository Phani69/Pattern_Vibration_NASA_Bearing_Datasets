# Pattern Vibration — NASA/IMS Bearing Datasets

Elevating predictive maintenance with vibration signal analytics and machine learning on NASA/IMS bearing run-to-failure datasets.

---

### TL;DR
- **Goal**: Detect incipient bearing faults and estimate remaining useful life (RUL) from raw vibration signals.
- **Data**: IMS/NASA bearing run-to-failure data pulled via the Kaggle API (slug: `vinayak123tyagi/bearing-dataset`).
- **Approach**: Signal processing (time/frequency/wavelet features) + classical ML baselines; room for DL extensions.
- **Artifacts**: Main analysis notebook `Predictive_maintanence.ipynb` (exploration → feature engineering → modeling → evaluation).

---

### Why this project?
Rotating machinery health monitoring is critical in aerospace and industrial applications. Vibration signals carry rich signatures of bearing degradation. This repo demonstrates a clean, end‑to‑end workflow for turning raw accelerometer data into actionable predictions for anomaly detection, fault diagnosis, and life estimation.

### Dataset (via Kaggle API)
This project uses the IMS run‑to‑failure bearing dataset obtained from Kaggle and attributed to the NASA/IMS sources. We fetch it programmatically using the Kaggle API. The notebook leverages `kagglehub` to download and cache the dataset locally.

What you typically get:
- Multiple test runs (R2F) with different operating conditions
- Raw time‑series vibration signals (often 20 kHz+ sampling)
- Ground truth “failure end” timestamps for supervised labeling or survival/RUL framing

If you don’t have the data locally yet, use one of the following options.

Option A — Python (recommended in this repo):
```python
import kagglehub

# Downloads and returns a local cache path like:
# ~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1
path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
print("Path to dataset files:", path)
```

Install if needed:
```bash
pip install kagglehub
# For private datasets or rate limits, configure Kaggle credentials as below
```

Option B — Kaggle CLI:
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/kaggle.json and restrict permissions
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo '{"username":"<your_username>","key":"<your_key>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download and unzip into a local data directory
mkdir -p data/IMS
kaggle datasets download -d vinayak123tyagi/bearing-dataset -p data/IMS --unzip
```

After downloading, you may organize as:

```
data/
  IMS/
    Run1/
    Run2/
    Run3/
```

Note: Folder names may vary based on your chosen download method (kagglehub cache vs. manual unzip). Adjust notebook paths accordingly. The notebook currently expects the kagglehub cache path similar to:

```
~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1/
```

### Repository structure
```
Pattern_Vibration_NASA_Bearing_Datasets/
  ├─ Predictive_maintanence.ipynb   # Main end‑to‑end analysis
  ├─ README.md                      # You are here
```

### Environment setup
Use Python 3.10+ and create an isolated environment. Example with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy scikit-learn pywavelets notebook jupyterlab tqdm
```

If you plan to experiment with deep learning models later, also consider:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### How to use
1) Ensure the dataset is available locally (see "Dataset").
2) Launch Jupyter and open the main notebook:
```bash
jupyter lab
```
3) Run cells top‑to‑bottom. The notebook covers:
- Data ingestion and sanity checks
- Signal conditioning (detrend, bandpass, resample)
- Feature engineering (time, frequency, wavelet, envelope)
- Health index construction and trend analysis
- Baseline models (classification for fault detection / regression for RUL proxies)
- Evaluation and visualization

### Reproducibility tips
- Fix seeds where applicable and log versions with:
```python
import sys, platform, numpy, pandas, sklearn
print(platform.python_version(), numpy.__version__, pandas.__version__, sklearn.__version__)
```
- Consider exporting a `requirements.txt` once your environment stabilizes:
```bash
pip freeze > requirements.txt
```

### Results (example directions)
This repo is designed as a strong starting point. Typical analyses include:
- Early fault detection ROC‑AUC and F1 across runs
- Degradation health index trends (monotonicity, smoothness)
- Proxy RUL regression error (MAE/RMSE) using engineered features

Feel free to extend with:
- 1D CNNs on raw or spectrogram inputs
- Transformers for long‑sequence modeling
- Self‑supervised pretext tasks (contrastive, masked prediction)

### Roadmap
- Add automated data loaders and configs for common IMS/NASA mirrors
- Package reusable feature extractors (`src/features/`)
- Add baseline DL models and benchmarking harness
- Publish lightweight web dashboard for real‑time trending

### Acknowledgments
- IMS (University of Cincinnati) for the bearing run‑to‑failure datasets
- NASA Prognostics Center of Excellence (PCoE) for maintaining the Prognostics Data Repository
---

Questions or ideas? Open an issue or start a discussion. Happy experimenting!
