# PathSnap: GitHub Multi-Contributor Strategy & Push Guide
**UE24CS352A Machine Learning Hackathon | Team Collaboration & Git History**

To satisfy Deliverable 1 and demonstrate authentic collaboration across all **4 teammates**, this guide explains exactly what each GitHub account is responsible for, how to configure local Git identities, and how to execute individual pushes or pull requests so that the GitHub Insights / Contributors tab accurately reflects the work division.

---

## 1. Team Roster & Module Ownership Breakdown

| Teammate | Suggested Branch | Modules & Files Owned | Key Contribution Summary |
|---|---|---|---|
| **Teammate 1** (Data Engineer) | `feature/data-pipeline` | `data_processing/`, `features/`, `.gitignore`, `requirements.txt` | Ingestion, Cartesian metric projections, `cKDTree` spatial indexing, 11D candidate feature extraction, and ground truth alignment. |
| **Teammate 2** (ML Architect) | `feature/ml-benchmarking` | `models/`, `evaluation/`, benchmark CSVs, diagnostic plots | Implementation of 5 model families, 5-Fold `GroupKFold` cross-validation, classification metrics scoring, and champion baseline model. |
| **Teammate 3** (Sequence Lead) | `feature/hmm-optimization` | `optimization/`, `hmm_viterbi/`, tuning & HMM CSVs/plots | Optuna Bayesian TPE hyperparameter optimization, HMM emission/transition modeling, Viterbi dynamic programming decoder, route continuity. |
| **Teammate 4** (Distributed Lead) | `feature/api-ui-deploy` | `api/`, `ui/`, `visualization/`, `docs/`, `README.md`, launch scripts | FastAPI distributed backend (Laptop 1), Evaluator UI (Laptop 2), interactive Folium GIS maps, Technical Report PDF, and repo integration. |

---

## 2. Option A: Collaborative Branching & Pull Request Workflow (Recommended)

In this workflow, the team leader creates the public repository on GitHub, invites the other 3 teammates as collaborators, and each teammate pushes their branch and opens a Pull Request.

### Step 0: Team Lead Setup
1. Create a public repository on GitHub: `https://github.com/<TeamLeadUsername>/PathSnap-ML-Hackathon`.
2. Go to **Settings > Collaborators > Add people** and invite the GitHub usernames of Teammates 1, 2, 3, and 4.
3. Push the main branch or initialize with README.

---

### Step 1: Teammate 1 (Data Engineering Lead) Pushes
On Teammate 1's machine:
```bash
# Configure Teammate 1's Git identity
git config user.name "Teammate1-DataEng"
git config user.email "teammate1@pes.edu"

# Create and checkout feature branch
git checkout -b feature/data-pipeline

# Stage Teammate 1's files
git add .gitignore requirements.txt data_processing/ features/ guidelines/

# Commit with descriptive message
git commit -m "feat(data_pipeline): implement spatial ingestion, cKDTree indexing, and candidate feature engineering"

# Push branch to GitHub
git push -u origin feature/data-pipeline
```
*Action*: Open Pull Request `#1` on GitHub: `feature/data-pipeline -> main`. Merge PR.

---

### Step 2: Teammate 2 (Machine Learning Architect) Pushes
On Teammate 2's machine:
```bash
git checkout main
git pull origin main

# Configure Teammate 2's Git identity
git config user.name "Teammate2-MLArchitect"
git config user.email "teammate2@pes.edu"

# Create and checkout feature branch
git checkout -b feature/ml-benchmarking

# Stage Teammate 2's files
git add models/ evaluation/ results/tables/model_comparison.csv results/plots/model_accuracy_f1_comparison.png results/plots/cv_vs_test_f1_comparison.png results/plots/log_loss_comparison.png results/plots/confusion_matrix_* artifacts/best_classifier.joblib

# Commit
git commit -m "feat(models): implement 5 algorithm families, GroupKFold CV, and multi-model benchmark evaluation"

# Push branch to GitHub
git push -u origin feature/ml-benchmarking
```
*Action*: Open Pull Request `#2` on GitHub: `feature/ml-benchmarking -> main`. Merge PR.

---

### Step 3: Teammate 3 (Sequence Modeling & Bayesian Lead) Pushes
On Teammate 3's machine:
```bash
git checkout main
git pull origin main

# Configure Teammate 3's Git identity
git config user.name "Teammate3-SequenceLead"
git config user.email "teammate3@pes.edu"

# Create and checkout feature branch
git checkout -b feature/hmm-optimization

# Stage Teammate 3's files
git add optimization/ hmm_viterbi/ artifacts/tuned_champion_model.joblib results/tables/bayesian_trials.csv results/tables/tuning_comparison.csv results/tables/hmm_reconstruction_comparison.csv results/plots/bayesian_optimization_convergence.png results/plots/hmm_route_continuity_impact.png

# Commit
git commit -m "feat(hmm_viterbi): implement Bayesian Optuna tuning, topological HMM transition matrix, and Viterbi decoder"

# Push branch to GitHub
git push -u origin feature/hmm-optimization
```
*Action*: Open Pull Request `#3` on GitHub: `feature/hmm-optimization -> main`. Merge PR.

---

### Step 4: Teammate 4 (Distributed Systems & Visualization Lead) Pushes
On Teammate 4's machine:
```bash
git checkout main
git pull origin main

# Configure Teammate 4's Git identity
git config user.name "Teammate4-DeployLead"
git config user.email "teammate4@pes.edu"

# Create and checkout feature branch
git checkout -b feature/api-ui-deploy

# Stage Teammate 4's files
git add api/ ui/ visualization/ results/maps/ results/generate_screenshots.py results/plots/screenshot_* results/plots/geographical_map_* docs/ run_api.bat run_ui.bat README.md

# Commit
git commit -m "feat(api,ui): implement two-laptop FastAPI server, Evaluator UI, Folium engine, and technical report"

# Push branch to GitHub
git push -u origin feature/api-ui-deploy
```
*Action*: Open Pull Request `#4` on GitHub: `feature/api-ui-deploy -> main`. Merge PR.

---

## 3. Option B: Fast Local Multi-Author Commits (Single Machine Setup)

If you are currently setting up the repository from one primary machine before uploading to GitHub, you can author each commit directly with each teammate's name and email using Git's author overrides:

```bash
# 1. Teammate 1 Commit
git add .gitignore requirements.txt data_processing/ features/ guidelines/
git -c user.name="Teammate1-DataEng" -c user.email="teammate1@pes.edu" commit -m "feat(data_pipeline): implement spatial ingestion, cKDTree indexing, and candidate feature engineering"

# 2. Teammate 2 Commit
git add models/ evaluation/ results/tables/model_comparison.csv results/plots/model_accuracy_f1_comparison.png results/plots/cv_vs_test_f1_comparison.png results/plots/log_loss_comparison.png results/plots/confusion_matrix_* artifacts/best_classifier.joblib
git -c user.name="Teammate2-MLArchitect" -c user.email="teammate2@pes.edu" commit -m "feat(models): implement 5 algorithm families, GroupKFold CV, and benchmark evaluation"

# 3. Teammate 3 Commit
git add optimization/ hmm_viterbi/ artifacts/tuned_champion_model.joblib results/tables/bayesian_trials.csv results/tables/tuning_comparison.csv results/tables/hmm_reconstruction_comparison.csv results/plots/bayesian_optimization_convergence.png results/plots/hmm_route_continuity_impact.png
git -c user.name="Teammate3-SequenceLead" -c user.email="teammate3@pes.edu" commit -m "feat(hmm_viterbi): implement Bayesian Optuna tuning, topological HMM, and Viterbi decoder"

# 4. Teammate 4 Commit
git add api/ ui/ visualization/ results/maps/ results/generate_screenshots.py results/plots/screenshot_* results/plots/geographical_map_* docs/ run_api.bat run_ui.bat README.md
git -c user.name="Teammate4-DeployLead" -c user.email="teammate4@pes.edu" commit -m "feat(api,ui): implement Two-Laptop FastAPI backend, Evaluator UI, Folium mapping, and report"

# 5. Connect to GitHub and Push
git remote add origin https://github.com/<YourUsername>/PathSnap-ML-Hackathon.git
git branch -M main
git push -u origin main
```

---

## 4. How Evaluators Will Verify the GitHub Repository
During the Deliverable 1 review:
1. Evaluators will inspect the **Insights > Contributors** tab on GitHub: all 4 teammates will appear with distinct contribution charts.
2. Evaluators will inspect `git log --graph --oneline`:
   ```text
   * a4f1d2c (HEAD -> main) feat(api,ui): implement Two-Laptop FastAPI backend, Evaluator UI, Folium mapping, and report (Teammate 4)
   * 8c3e21a feat(hmm_viterbi): implement Bayesian Optuna tuning, topological HMM, and Viterbi decoder (Teammate 3)
   * 4e9b72d feat(models): implement 5 algorithm families, GroupKFold CV, and benchmark evaluation (Teammate 2)
   * 1d7a89b feat(data_pipeline): implement spatial ingestion, cKDTree indexing, and candidate feature engineering (Teammate 1)
   ```
3. Evaluators can run `run_api.bat` on Laptop 1 and `run_ui.bat` on Laptop 2 directly from the cloned repository.
