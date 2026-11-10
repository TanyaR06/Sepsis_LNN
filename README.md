# TA-LNN / TALNet — Topology-Aware Liquid Neural Network for Sepsis Early Warning

A full, deployable implementation of the system in the paper *"Topology-Aware Liquid
Neural Networks for Early Prediction of Sepsis-Induced Multi-Organ Failure from Irregularly
Sampled ICU Streams."*

It ingests irregular PhysioNet/CinC-2019 ICU streams **without imputation**, computes
**per-organ persistent-homology descriptors**, runs a **coupled Liquid Time-Constant network**,
produces an **organ-cascade attribution**, and wraps the alarm in a **split-conformal calibration**
layer. A polished clinical dashboard lets you select or upload a patient and read the early-warning
risk trajectory, calibrated alarm, organ-wise contribution, confidence, and lead time.

The bundled weights were trained on **real PhysioNet/CinC 2019 Set A** data.

---

## 1. What maps to what (paper → code)

| Paper component (Section) | Equation | File |
|---|---|---|
| Irregular representation `u_t = [x⊙m, m, Δ]` (III-A) | Eq. 1 | `backend/data.py` |
| Per-organ point cloud + persistent homology → `φ_o(t)` (III-B) | Eq. 2–3 | `backend/topology.py` |
| Coupled liquid dynamics, per-organ `τ_base` (III-C) | Eq. 4–6 | `backend/model.py` |
| Risk head + organ-cascade head (III-C) | Eq. 7–8 | `backend/model.py` |
| Split-conformal alarm calibration (III-D) | — | `backend/conformal.py` |
| Metrics: AUROC/AUPRC/lead time/FAR/coverage (IV-C) | Eq. 9–11 | `backend/metrics.py` |
| Training + inference procedure (Table 1) | — | `backend/train.py`, `backend/inference.py` |

The six organ subsystems (respiratory, cardiovascular, renal, hepatic, coagulation, metabolic)
and their physiological base time-constants are defined in `backend/config.py`.

---

## 2. Folder structure

```
talnet/
├── backend/
│   ├── config.py          # channel layout, organ partition, τ_base, hyper-params
│   ├── data.py            # PSV loader, mask/gap representation, scaler, organ targets
│   ├── topology.py        # per-organ persistent homology (ripser + MST fallback)
│   ├── model.py           # coupled Liquid Time-Constant network + heads (autograd)
│   ├── conformal.py       # split-conformal calibration + FAR-controlled operating point
│   ├── metrics.py         # AUROC / AUPRC / lead time / FAR / coverage
│   ├── train.py           # end-to-end training, calibration, evaluation, export
│   ├── inference.py       # single-record inference API (TALNet class)
│   └── app.py             # FastAPI service + dashboard host
├── frontend/
│   └── index.html         # single-file clinical dashboard (Chart.js)
├── models/                # trained artifacts (shipped)
│   ├── talnet_weights.npz # trained TA-LNN weights
│   ├── scaler.json        # standardization stats (observed values only)
│   ├── conformal.json     # calibrated thresholds + risk distributions
│   ├── meta.json          # metrics, class distribution, paper reference values
│   └── demo_patients.json # precomputed held-out demo trajectories
├── data/demo/             # sample PhysioNet .psv records for upload testing
├── scripts/
│   ├── run_server.sh      # launch API + dashboard
│   └── train.sh           # retrain on your own PhysioNet folder
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 3. Quick start (local)

```bash
cd talnet
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# launch the dashboard + API
uvicorn backend.app:app --host 0.0.0.0 --port 8000
#   or:  PORT=8000 ./scripts/run_server.sh
```

Open **http://localhost:8000** in a browser. The dashboard has five tabs:

- **Patient Risk Explorer** — pick a held-out patient; see the hour-by-hour risk, the conformal
  alarm firing, the labelled onset, and the resulting early-warning lead time.
- **Organ-Cascade** — a heatmap of which subsystem drives the risk at each hour, plus the
  per-organ effective time constants `τ_o`.
- **Upload & Predict** — drop in any PhysioNet `.psv` record and get a live analysis.
- **Live Vitals** — a slider-driven what-if that scores a 24-hour trajectory in real time.
- **Model Dashboard** — this run's metrics vs the paper, baselines, ablations, and class balance.

> If `ripser` is not installed, topology gracefully falls back to an exact H0/MST descriptor
> (H1 loop features set to 0); everything else is unchanged.

---

## 4. Run with Docker

```bash
docker compose up --build          # serves on http://localhost:8000
# or
docker build -t talnet:1.0.0 .
docker run -p 8000:8000 talnet:1.0.0
```

---

## 5. REST API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + whether weights are present |
| GET | `/api/meta` | metrics, param count, operating point, paper reference |
| GET | `/api/demo` | list of demo patients |
| GET | `/api/demo/{id}` | full trajectory (risk, cascade, τ) for one demo patient |
| POST | `/api/predict` | multipart upload of a `.psv` → full inference JSON |
| POST | `/api/predict_json` | inference on a JSON record `{columns, data}` |
| GET | `/api/sample.psv` | download a sample record for testing |

Example:

```bash
curl -F "file=@data/demo/p008586.psv" http://localhost:8000/api/predict
```

Response (abridged):

```json
{
  "n_hours": 77, "risk": [...], "alarm": [...], "threshold": 0.31,
  "cascade": [[...6 organs...], ...], "primary_driver_label": "Cardiovascular",
  "onset_hour": 69, "first_alarm_hour": 68, "lead_time_hours": 1,
  "confidence": 0.80, "false_alarm_pvalue": 0.02, "coverage_level": 0.9
}
```

---

## 6. Retraining on your own PhysioNet folder

```bash
DATA=/path/to/training_setA EPOCHS=22 ./scripts/train.sh
# or directly:
python -m backend.train --data /path/to/training_setA \
    --n-train 1000 --n-cal 400 --n-test 700 --epochs 22 \
    --pos-weight 8 --target-far 0.10 --procs 2
```

Training is **resumable**: it caches the (expensive) topology features to
`models/_feature_cache.pkl` and checkpoints every epoch to `models/_train_ckpt.npz`, so it can
be stopped and restarted (`--max-seconds N` pauses gracefully). Re-running the same command
resumes automatically. Artifacts are written to `models/`.

To evaluate across institutions as the paper does (train on cohort A, test on cohort B), point
`--data` at Set A and add a second held-out folder — the split logic in `train.py` is the place
to wire in an external test cohort.

---

## 7. Results (this bundled run vs the paper)

Trained on a patient-level split of **PhysioNet/CinC 2019 Set A** (1000 train / 400 calibration /
700 test), evaluated on the held-out test split.

| Metric | This run | Paper TA-LNN |
|---|---|---|
| AUROC | 0.76 | 0.934 |
| AUPRC | 0.13 | 0.648 |
| Median lead time @ FAR 0.10 | ~27 h | 7.4 h |
| False-alarm rate | 0.08 | 0.091 |
| Conformal coverage | 90.2 % | 90.2 % |
| Trainable parameters | ~17 k | ~790 k |

**Honesty notes.** This deployment trains a deliberately *compact* TA-LNN (~17 k parameters) on a
patient-level split of Set A only (Set B was not available), for a few dozen epochs on CPU. The
paper's headline numbers come from a larger-capacity configuration trained on cohort A and tested
on the independent cohort B. The conformal coverage guarantee reproduces at the 90 % level, and the
alarm's operating threshold is placed at a fixed patient-level false-alarm rate — the same protocol
the paper uses to report lead time. The **entire pipeline is faithful to the methodology**: no
imputation, real per-organ persistent homology, coupled liquid units with per-organ time constants,
organ-cascade attribution, and split-conformal calibration. Increase `--n-train`, `HIDDEN_PER_ORGAN`
(in `config.py`), and `--epochs` to push discrimination toward the paper's figures.

---

## 8. Clinical-use disclaimer

This is a **research and demonstration** system built for reproducing a methods paper and for an
IEEE competition demo. It is **not** a medical device and must not be used for clinical decisions.
