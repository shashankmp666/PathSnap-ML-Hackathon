# PathSnap: Comprehensive Viva & Technical Defense Guide
**UE24CS352A: Machine Learning Hackathon | PES University**

This document serves as the master technical defense and question-and-answer preparation sheet for all four team members during the viva demonstration before the evaluators and faculty.

---

## 1. Teammate 1 Defense: Data Engineering & Spatial Preprocessing Specialist

### Q1.1: What are the raw files provided in each trip, and how do their data structures interrelate?
- **Answer**:
  1. `.track`: Whitespace-separated text with 3 columns: `longitude`, `latitude`, `timestamp` in seconds. Sampled at approximately 1 Hz (about 98.1% of intervals are exactly 1.0s apart).
  2. `.nodes`: 2 columns: `longitude`, `latitude`. The 0-based row index defines the `node_id`. Represents physical intersections and geometric road curve vertices.
  3. `.arcs`: 2 integer columns: `from_node_id`, `to_node_id`. The 0-based row index defines the directed `arc_id`.
  4. `.route`: 1 column of integer `arc_id` values traversed chronologically by the vehicle.
- **Key Insight**: A row in `.track` does **not** correspond 1-to-1 with a row in `.arcs`. Vehicles travel on a single arc for several consecutive seconds, meaning multiple GPS track points lie on the same arc.

### Q1.2: Why can we not use ordinary Euclidean distance directly on (lon, lat) degrees?
- **Answer**: Longitude and latitude are spherical angular coordinates on an oblate spheroid (WGS84).
  - 1 degree of latitude is roughly constant worldwide ($\approx 111,320$ meters).
  - 1 degree of longitude scales with the cosine of latitude: $\Delta x = \Delta \text{lon} \times 111,320 \times \cos(\text{lat})$.
  - At the dataset's latitude ($\approx 49.85^\circ$ N around Darmstadt, Germany), $\cos(49.85^\circ) \approx 0.6448$, meaning 1 degree of longitude is only $\approx 71,780$ meters.
  - Using unweighted Euclidean distance in degrees would compress longitudinal distances by $\approx 35.5\%$, severely distorting orthogonal projections, angles, and distances.
  - **Solution**: We implemented local equirectangular Cartesian projection converting all coordinates to metric Cartesian meters $(x, y)$ centered at the trajectory reference latitude.

### Q1.3: How does your candidate generation work, and what is the candidate miss rate?
- **Answer**:
  - We build a 2D spatial tree (`scipy.spatial.cKDTree`) on the metric midpoints of all directed road segments in the trip's road network.
  - For each GPS observation $t$, we retrieve the nearest 60 arc midpoints, compute exact orthogonal segment projections, and rank them by perpendicular distance.
  - We retain the top $k=10$ closest candidates per observation.
  - **Empirical Candidate Miss Rate**: Exactly **3.31%** on the training partition and **2.91%** on the student-test partition (meaning candidate retrieval recall exceeds **97.0%**!).

### Q1.4: How are candidate features calculated?
- **Answer**: For each candidate pair $(z_t, c_i)$, we extract an 11-dimensional feature vector:
  1. `perp_dist`: Orthogonal distance from $(x_t, y_t)$ to directed segment $\mathbf{a} \to \mathbf{b}$ in meters.
  2. `proj_ratio`: Projection parameter $t_{\text{proj}} = \frac{(\mathbf{p}_z - \mathbf{a}) \cdot (\mathbf{b} - \mathbf{a})}{\|\mathbf{b} - \mathbf{a}\|^2}$ clamped to $[0.0, 1.0]$.
  3. `dist_to_start` / `dist_to_end`: Distances to segment terminal nodes.
  4. `arc_length`: Physical segment length $\|\mathbf{b} - \mathbf{a}\|$.
  5. `heading_diff`: Minimum angular difference $[0^\circ, 180^\circ]$ between vehicle velocity vector and arc orientation vector.
  6. `cos_heading_diff`: Cosine of heading difference $[-1.0, 1.0]$.
  7. `speed_mps`: Instantaneous GPS speed.
  8. `time_delta`: $\Delta \tau = \tau_t - \tau_{t-1}$.
  9. `dist_rank`: Discrete ranking ($0, 1, \dots, 9$) among candidates for observation $t$.
  10. `gaussian_prior`: Gaussian measurement error $\exp(-d_{\perp}^2 / (2\sigma_z^2))$ with $\sigma_z = 10\text{ m}$.

---

## 2. Teammate 2 Defense: Machine Learning Architect & Benchmark Lead

### Q2.1: Which 5 distinct machine learning algorithm families did you train and compare?
- **Answer**:
  1. **Linear / Probabilistic**: Logistic Regression with $L_2$ regularization (`StandardScaler` + `LogisticRegression(C=1.0, max_iter=1000)`).
  2. **Instance-Based**: Distance-Weighted $K$-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=11, weights='distance')`).
  3. **Tree-Based**: Single Decision Tree (`DecisionTreeClassifier(max_depth=10, min_samples_leaf=20)`).
  4. **Ensemble**: Random Forest Classifier (`RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=5)`).
  5. **Gradient Boosting**: Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31)`).

### Q2.2: How did you ensure zero data leakage during cross-validation?
- **Answer**:
  - We used `GroupKFold(n_splits=5)` grouped strictly by `trip_id`.
  - Standard $K$-Fold would leak observations from the same vehicle trip across training and validation folds, causing unrealistically inflated scores due to spatial autocorrelation.
  - `GroupKFold` guarantees that all candidate rows from any given trip are strictly isolated to either the train fold or the validation fold, ensuring true out-of-trip generalization.

### Q2.3: Summarize your experimental benchmark findings. Why was the champion model selected?
- **Answer**:
  - **Random Forest** achieved the highest 5-Fold CV Macro F1 (**0.8506**) and student-test Top-1 Accuracy (**79.21%**), with the lowest test Log Loss (**0.1033**).
  - **HistGradientBoosting** achieved nearly identical performance (**79.09%** Top-1 Accuracy, **0.8489** CV Macro F1, **0.1063** Log Loss) while training in half the time (**1.22s** vs **2.40s**) and producing calibrated probabilities.
  - **Logistic Regression** produced a respectable 78.30% Top-1 accuracy but struggled with nonlinear interactions between velocity, heading, and distance.
  - **KNN** had the lowest Top-1 accuracy (77.29%) and the highest inference latency (0.0290 ms/cand vs 0.0004 ms/cand for tree ensembles) due to scanning stored training points in high dimensions.
  - **HistGradientBoosting / Random Forest** were chosen for their robust generalization, fast inference, and smooth probability calibration.

---

## 3. Teammate 3 Defense: Sequence Modeling & Bayesian Optimization Lead

### Q3.1: How did you design the Bayesian hyperparameter optimization search?
- **Answer**:
  - **Framework**: Optuna using Tree-structured Parzen Estimator (TPE) sampler.
  - **Target Model**: `HistGradientBoostingClassifier`.
  - **Objective Function**: Mean 5-Fold `GroupKFold` Top-1 Accuracy across training trips.
  - **Search Space**:
    - `learning_rate`: Log-uniform $\in [0.02, 0.25]$
    - `max_leaf_nodes`: Integer $\in [15, 63]$
    - `min_samples_leaf`: Integer $\in [10, 80]$
    - `l2_regularization`: Log-uniform $\in [10^{-4}, 5.0]$
    - `max_iter`: Integer $\in [60, 140]$ with step 20
  - **Trials**: 20 optimization trials.
  - **Best Configuration Obtained**: `learning_rate` = 0.02038, `max_leaf_nodes` = 56, `min_samples_leaf` = 44, `l2_reg` = 0.1813, `max_iter` = 100.
  - **Outcome**: Tuned Top-1 accuracy improved to **79.22%**, with improved validation stability across folds.

### Q3.2: How is the Hidden Markov Model (HMM) formulated for route reconstruction?
- **Answer**:
  - **Hidden States**: At each time step $t$, the state space $S_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, k}\}$ consists of the $k=10$ candidate road segments for observation $z_t$.
  - **Emission Probability $P(z_t | c_i)$**:
    $$\ln P_{\text{emit}} = \alpha \ln \max(P_{\text{model}}(c_i), 10^{-4}) + (1-\alpha) \left(-\frac{d_{\perp}(z_t, c_i)^2}{2\sigma_z^2}\right)$$
    where $\sigma_z = 10\text{ m}$ is the GPS measurement noise std, $P_{\text{model}}$ is the classifier confidence, and $\alpha = 0.85$.
  - **Transition Probability $P(c_{t, j} | c_{t-1, i})$**:
    - If $c_{t-1, i} == c_{t, j}$ (same road segment): $\ln P_{\text{trans}} = +3.0$ (continuity bonus).
    - If $c_{t, j}$ directly succeeds $c_{t-1, i}$ in the road network (`to_node(i) == from_node(j)`): $\ln P_{\text{trans}} = +3.5$ (topological link bonus).
    - If disconnected: $\ln P_{\text{trans}} = -15.0$ (severe penalty for non-physical teleportation).

### Q3.3: How does the Viterbi Dynamic Programming algorithm find the optimal path?
- **Answer**:
  - Forward recursion maintains $V[t, j]$: the maximum log-probability of reaching state $j$ at step $t$:
    $$V[t, j] = \max_{i} \left( V[t-1, i] + \ln P_{\text{trans}}(c_j | c_i) \right) + \ln P_{\text{emit}}(z_t | c_j)$$
  - We store the argmax index in a backpointer matrix $B[t, j] = \text{argmax}_i (\dots)$.
  - After processing all $T$ observations, we pick the global maximum state $s_T^* = \text{argmax}_j V[T-1, j]$ and trace backpointers from $T-1$ down to $0$ to reconstruct the globally optimal route sequence.

### Q3.4: Did HMM decoding improve the final route?
- **Answer**:
  - **Yes, dramatically on topological continuity**: Route continuity jumped from **89.27%** (isolated classifier argmax) to **99.71%** (HMM Viterbi), an absolute gain of **+10.44%**. In 3 of the 10 test trips (007, 040, 077), HMM achieved **100.0% perfect continuity**.
  - **Observation-level accuracy trade-off**: Isolated accuracy moved slightly from 80.99% to 79.77% (-1.21%). This occurs because noisy GPS points near intersections or highway overpasses often lie 2 meters closer to an unrelated side street or ramp; the isolated classifier blindly picks that side street, causing an impossible jump. The HMM regularizes the path to remain on the connected highway corridor, preserving true vehicle physics.

---

## 4. Teammate 4 Defense: Distributed Systems, API Backend & Visualization Lead

### Q4.1: Explain the Two-Laptop Deployment Architecture and how data isolation is enforced.
- **Answer**:
  - **Laptop 1 (Model Host & FastAPI Server)**:
    - Hosts the trained classifier weights (`tuned_champion_model.joblib`), HMM Viterbi engine, road-network graph, and server-side evaluation pipeline.
    - Runs on port 8000 accessible across the local network via its LAN IP address.
    - Exposes endpoints `/health`, `/trips`, `/predict/{trip_id}`, `/evaluate/{trip_id}`, and `/maps/{trip_id}`.
    - **Zero Data Leakage**: Ground-truth `.route` files remain strictly inside Laptop 1's local filesystem. The `/evaluate` endpoint computes metrics server-side and returns only numeric scores and rendered map links. Ground truth is never sent to the UI.
    - Outputs live evaluation audit logs on the CMD console for faculty inspection.
  - **Laptop 2 (Evaluator User Interface)**:
    - Lightweight web dashboard with zero backend dependencies.
    - Displays the 10 reserved evaluation trips in a responsive dropdown.
    - Transmits only the selected `trip_id` via HTTP POST to Laptop 1.
    - Renders the returned accuracy, route continuity, and latency scoreboard, and embeds the interactive Folium map in an iframe.

### Q4.2: How is the Folium interactive map constructed, and what layers are supported?
- **Answer**:
  - Built dynamically per trip using `folium.Map` with OpenStreetMap base tiles.
  - Features 4 distinct visual layers managed by `folium.LayerControl`:
    1. **Road Network Geometry** (Gray lines, opacity 0.6): Visualizes all local road segments within the trip bounding box.
    2. **Ground Truth Route** (Emerald Green polyline, width 5.0): Shows the true sequence of traversed arcs.
    3. **Raw GPS Trajectory** (Blue dashed line + blue circle markers with popups): Shows raw, noisy 1 Hz sensor points with timestamp and speed.
    4. **PathSnap Predicted Route** (Crimson red polyline, width 4.0): Visualizes the reconstructed path produced by HMM Viterbi.
  - Includes a floating CSS map legend and interactive zoom/pan controls.

### Q4.3: How would you demonstrate the live system to the evaluator right now?
- **Answer**:
  1. Show Laptop 1's terminal running `python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`.
  2. Open Laptop 2's browser to `ui/index.html`.
  3. Show the green `Connected to Laptop 1` badge.
  4. Ask the evaluator to select any of the 10 trips (e.g. Trip `00000052`).
  5. Click `Run Evaluation on Laptop 1`.
  6. Point to Laptop 1's CMD terminal showing the live audit log:
     ```text
     [EVALUATION AUDIT LOG] Trip ID: 00000052
     Client IP:                127.0.0.1
     Observations Matched:     262
     Traversed Arcs Decoded:   90
     Observation Top-1 Acc:    93.13%
     Route Continuity:         99.62%
     Candidate Miss Rate:      0.76%
     Total Inference Latency:  1041.0ms
     ```
  7. Show Laptop 2's UI displaying the matching metrics and the rendered Folium map showing the green and red paths aligning along the highway.
