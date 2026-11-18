# PathSnap

Map-matching and sequential route reconstruction for noisy GPS trajectories on directed road networks. Combines spatial candidate retrieval (`cKDTree`), kinematic feature extraction, supervised gradient boosting, and topological Hidden Markov Model (HMM) Viterbi decoding to reconstruct vehicle routes with high observation accuracy and continuous network connectivity.

---

## Overview

In urban environments, raw GPS sensor data exhibits significant spatial noise due to multipath reflection, tall buildings, and discrete sampling intervals (typically 1 Hz). Unconstrained pointwise matching often leads to unrealistic jumps between parallel lanes, opposing traffic corridors, or disconnected road segments.

PathSnap solves this through a two-stage formulation:
1. **Candidate Retrieval & Scoring**: For each GPS point, nearby road segments are retrieved using a 2D spatial `cKDTree`. An 11-dimensional geometric and kinematic feature vector (perpendicular distance, heading delta, along-track projection, speed consistency) is scored by a trained Gradient Boosted Decision Ensemble.
2. **Sequential Route Decoding**: Candidate emission scores are combined with road network topological transition matrices in an HMM Viterbi decoder. Reverse-lane transitions and disconnected hops are heavily penalized, reconstructing a continuous directed trajectory through the road network.

---

## Benchmark Results

### Official Evaluator Test Benchmark (10 Unseen Trips)

Evaluated on the official `evaluator_test` partition containing 10 trajectories (19,674 GPS observations). PathSnap achieves **> 92% Top-1 Accuracy across all 10 evaluation trips**, with **100% Candidate Recall** and **99.49% Route Continuity**:

| Trip ID | Observations | Candidate Recall | Candidate Miss Rate | Classification Acc | Weighted F1 | Top-1 Acc (HMM) | Route Continuity | Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000015` | 6,854 | 100.0% | 0.0% | 90.38% | 89.83% | **99.72%** | 100.00% | 4.4 s |
| `00000017` | 1,346 | 100.0% | 0.0% | 95.46% | 95.20% | **92.79%** | 98.14% | 3.5 s |
| `00000023` | 458 | 100.0% | 0.0% | 98.62% | 98.60% | **98.25%** | 100.00% | 0.4 s |
| `00000030` | 654 | 100.0% | 0.0% | 93.88% | 93.60% | **93.88%** | 98.62% | 0.9 s |
| `00000049` | 2,642 | 100.0% | 0.0% | 98.91% | 98.91% | **97.73%** | 100.00% | 4.1 s |
| `00000059` | 1,031 | 100.0% | 0.0% | 95.90% | 95.67% | **94.37%** | 98.64% | 1.7 s |
| `00000067` | 916 | 100.0% | 0.0% | 95.62% | 95.49% | **98.80%** | 100.00% | 0.6 s |
| `00000073` | 1,144 | 100.0% | 0.0% | 97.32% | 97.23% | **97.03%** | 99.83% | 1.2 s |
| `00000087` | 1,399 | 100.0% | 0.0% | 97.91% | 97.91% | **95.28%** | 99.86% | 2.5 s |
| `00000094` | 3,230 | 100.0% | 0.0% | 94.37% | 93.96% | **97.31%** | 99.85% | 6.7 s |
| **OVERALL** | **19,674** | **100.0%** | **0.0%** | **94.31%** | **94.07%** | **96.52%** | **99.49%** | **2.6 s** |

### Multi-Model Comparison (5 Algorithm Families)

Models evaluated across 5-fold `GroupKFold` cross-validation (isolated by `trip_id`):

| Model | Family | 5-Fold CV Macro F1 | Test Top-1 Acc | Route Continuity | Test Macro F1 | Log Loss | Latency (ms/cand) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Linear / Probabilistic | 0.8330 | 78.30% | 88.78% | 0.8653 | 0.1309 | 0.0001 ms |
| **KNN (Distance-Weighted)** | Instance-Based ($k=11$) | 0.8309 | 77.29% | 88.09% | 0.8657 | 0.1876 | 0.0290 ms |
| **Decision Tree** | Tree-Based | 0.8416 | 78.55% | 89.11% | 0.8797 | 0.1163 | 0.0001 ms |
| **Random Forest** | Bagging Ensemble (80 trees) | **0.8506** | **79.21%** | **89.70%** | **0.8842** | **0.1033** | 0.0005 ms |
| **HistGradientBoosting** | Gradient Boosting | 0.8489 | 79.09% | 89.21% | 0.8827 | 0.1063 | 0.0004 ms |

### Bayesian Optimization (Optuna TPE)

`HistGradientBoostingClassifier` was tuned over 20 trials using Tree-structured Parzen Estimators (TPE) on 5-fold `GroupKFold` CV:
- Best parameters: `learning_rate` = 0.02038, `max_leaf_nodes` = 56, `min_samples_leaf` = 44, `l2_regularization` = 0.1813, `max_iter` = 100.
- Outperformed untuned baseline while improving probability calibration for HMM emissions.

### HMM Viterbi Sequential Recovery

| Metric | Pointwise Classifier | HMM Viterbi Decoding | Impact |
|---|:---:|:---:|---|
| **Route Continuity** | 89.27% | **99.71%** | +10.44% (eliminates disconnected hops) |
| **Trips with 100% Continuity** | 0 / 10 | **4 / 10** | Resolves opposing-lane & overpass jitter |

---

## Team Attribution

| Teammate | Focus Area | Modules | Responsibilities |
|---|---|---|---|
| **Teammate 1** | Data Engineering | `data_processing/`, `features/` | Trajectory ingestion, Cartesian metric transformations, spatial indexing (`cKDTree`), candidate extraction, and ground-truth sequence alignment. |
| **Teammate 2** | Machine Learning | `models/`, `evaluation/` | Model training across 5 families, 5-fold `GroupKFold` CV pipeline, evaluation metrics, and comparative benchmarking. |
| **Teammate 3** | Sequential Modeling | `optimization/`, `hmm_viterbi/` | Bayesian hyperparameter optimization (Optuna TPE), HMM emission/transition calibration, and Viterbi dynamic programming decoder. |
| **Teammate 4** | Distributed Systems & UI | `api/`, `ui/`, `visualization/` | FastAPI server endpoints (`/predict`, `/evaluate`), server-side ground truth isolation, Evaluator web interface, and interactive Folium GIS mapping. |

---

## Project Structure

```text
├── api/
│   ├── app.py                      # FastAPI server application
│   └── schemas.py                  # Pydantic response/request schemas
├── artifacts/
│   ├── best_classifier.joblib      # Baseline champion classifier
│   └── tuned_champion_model.joblib # Bayesian-tuned champion model
├── data/
│   ├── train/                      # Training trajectories (.track, .nodes, .arcs, .route)
│   ├── student_test/               # Student test trajectories
│   └── evaluator_test/             # 10 reserved evaluation trajectories
├── data_processing/
│   ├── dataset.py                  # Dataset loader & road network graph builder
│   └── spatial_utils.py            # Coordinate projections and azimuth calculations
├── features/
│   ├── feature_extractor.py        # 11D feature extractor & candidate generation
│   └── build_dataset.py            # Feature dataset builder
├── hmm_viterbi/
│   └── hmm_decoder.py              # RoadNetworkGraph and HMM Viterbi decoder
├── models/
│   └── train_all_models.py         # Multi-model training and benchmarking script
├── optimization/
│   └── bayesian_optimization.py    # Optuna TPE tuning script
├── results/
│   ├── maps/                       # Interactive Folium HTML trajectory maps
│   └── tables/                     # Benchmark CSV output tables
├── ui/
│   └── index.html                  # Minimalist Evaluator web interface
├── docs/
│   ├── PathSnap_Technical_Report.pdf # Consolidated technical report
│   └── VIVA_DEFENSE_GUIDE.md       # Viva defense Q&A preparation guide
├── evaluate_evaluator_test.py      # Standalone evaluator benchmark script
├── run_api.bat                     # Server startup script
└── run_ui.bat                      # UI launcher script
```
