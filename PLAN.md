# Diabetic Retinopathy Stage Detection — Coursework Plan

Deliverables: (1) commented codebase, (2) PDF report <= 20 pages, (3) hosted demo video URL in report.
Marks in brackets map to rubric. Do tasks in order — each builds on previous.

---

## Phase 0 — Setup (no marks, but everything depends on it)

- [x] 0.1 Compute: **Ubuntu PC, RTX 3060 12GB** (see dr_project/SETUP_UBUNTU_GPU.md). Kaggle = fallback. Windows laptop = CPU-only dev + report.
- [x] 0.2 Identify exact Kaggle dataset slug for `archive/` (35,126 imgs, 5 folders) so it can be pulled via Kaggle API in cloud — avoids uploading 35k images.
- [x] 0.3 Create repo structure:
      ```
      dr_project/
        01_eda.ipynb            # dataset exploration
        02_preprocessing.ipynb  # preprocessing pipeline + before/after figs
        03_augmentation.ipynb   # augmentation + balancing evidence
        04_train.ipynb          # model, training
        05_evaluate.ipynb       # metrics, curves, confusion matrix, Grad-CAM
        app/  (demo: Gradio/Streamlit for video)
        src/  (reusable .py modules: preprocess.py, data.py, model.py, utils.py)
        figures/                # every plot saved here → report
        README.md
      ```
- [x] 0.4 Fix seeds, pin library versions (`requirements.txt`) → reproducibility (rubric 7).

## Phase 1 — Problem Understanding & Dataset Justification [10]

- [ ] 1.1 Write-up: what DR is, 5 ICDR stages (No DR → Mild NPDR → Moderate → Severe → PDR), lesions per stage (microaneurysms, haemorrhages, exudates, neovascularisation), why screening matters.
- [x] 1.2 EDA notebook: class counts bar chart, sample grid per class, image size/aspect stats, left/right eye distribution from `trainLabels.csv`.
- [x] 1.3 Stratified split train/val/test (e.g. 70/15/15). **Split by patient ID** (`10_left`,`10_right` = same patient) → no leakage. Save split CSVs.
- [ ] 1.4 Note limitations: label noise, imbalance, single-source camera, resized images lose fine lesions. Ethics: patient data, bias, not a diagnostic device.

## Phase 2 — Preprocessing [10]

- [x] 2.1 Implement in `src/preprocess.py`: crop black borders (threshold mask) → resize to fixed square (224 or 256) → circular mask.
- [x] 2.2 Contrast enhancement: CLAHE on green channel / LAB L-channel. Compare with Ben Graham method (Gaussian-blur subtraction). Pick one, justify.
- [x] 2.3 Noise removal: light Gaussian/median blur; edge enhancement (unsharp mask) — show effect on vessels/lesions.
- [x] 2.4 Normalisation (ImageNet mean/std, since transfer learning).
- [x] 2.5 Figure: before/after grid for each step, per class. Histogram of intensities before/after CLAHE.
- [x] 2.6 Optional: pre-process whole dataset once to disk (speeds training massively).

## Phase 3 — Augmentation & Balancing [10]

- [x] 3.1 Augmentations (retina-safe): rotation (any angle — fundus is rotation-invariant), h/v flip, small zoom, brightness/contrast jitter. Avoid heavy shear/colour shift (changes lesion look).
- [x] 3.2 Visualise: one image → 8 augmented versions.
- [x] 3.3 Imbalance: choose + justify. Recommended: class-weighted loss **and** oversampling minority classes in sampler. Show class distribution before/after.
- [ ] 3.4 (done in Phase 4 training) Ablation: train few epochs with vs without balancing → table for report.

## Phase 4 — CNN Architecture & Transfer Learning [20] ← biggest chunk

- [ ] 4.1 Baseline: small custom CNN from scratch (few epochs) → proves transfer learning helps.
- [ ] 4.2 Main model: pretrained ImageNet backbone (EfficientNet-B0/B3 or ResNet50) + GAP + Dropout + Dense(5).
- [ ] 4.3 Two-stage: (a) freeze backbone, train head; (b) unfreeze top N blocks, low LR fine-tune.
- [ ] 4.4 Compare >= 2 backbones (e.g. ResNet50 vs EfficientNet) → table: params, val acc, F1, train time.
- [ ] 4.5 Hyperparameter tuning: LR, dropout, image size, batch size — record in table. Justify final choice.
- [ ] 4.6 Diagram of final architecture for report.

## Phase 5 — Training Strategy [10]

- [ ] 5.1 Callbacks: EarlyStopping (val loss), ReduceLROnPlateau or cosine schedule, ModelCheckpoint (best val F1/kappa).
- [ ] 5.2 Overfitting control: dropout, augmentation, weight decay, label smoothing (optional).
- [ ] 5.3 Log every run (CSV logger / W&B) → experiment table in report.
- [ ] 5.4 Save `history` for curves + final weights (.keras / .pt).

## Phase 6 — Evaluation [15]

- [ ] 6.1 Accuracy + loss curves (train vs val), both phases on one plot.
- [ ] 6.2 Test set: accuracy, per-class precision/recall/F1, macro + weighted F1, confusion matrix (raw + normalised).
- [ ] 6.3 Extra (DR-standard): Quadratic Weighted Kappa, ROC-AUC per class, binary "referable DR" (Moderate+) sensitivity/specificity.
- [ ] 6.4 Error analysis: show misclassified examples, discuss adjacent-stage confusion (Mild↔Moderate).
- [ ] 6.5 Grad-CAM heatmaps → shows model looks at lesions, not artefacts.

## Phase 7 — Code Quality [10]

- [ ] 7.1 Move logic into `src/*.py`, notebooks call them. Docstrings + comments on every function.
- [ ] 7.2 README: setup, how to run each step, where outputs go.
- [ ] 7.3 Clean rerun of all notebooks top-to-bottom before submission.

## Phase 8 — Demo App + Video [part of 8]

- [ ] 8.1 Gradio/Streamlit app: upload fundus image → preprocessed preview → predicted stage + probabilities + Grad-CAM.
- [ ] 8.2 Record 2–4 min screen video: dataset → preprocessing → training curves → app inference.
- [ ] 8.3 Upload (YouTube unlisted / Google Drive), put URL in report.

## Phase 9 — Report (<= 20 pages) [10 + 5]

- [ ] 9.1 Structure: Title → Intro & DR background → Dataset → Preprocessing → Augmentation/Balancing → Model & Transfer Learning → Training → Results → Discussion (impact, deployment, limits, ethics, future work) → Conclusion → References → Appendix (video URL, repo link).
- [ ] 9.2 Every section = figure + explanation (rubric wants "graphical evidence" for all steps).
- [ ] 9.3 Map each section to LO1–LO4 explicitly.
- [ ] 9.4 Innovation/critical discussion [5]: real clinic use, edge deployment, model limits, dataset bias, future (higher res, ordinal loss, ensembles).
- [ ] 9.5 Page-count check, export PDF.

---

## Suggested order of work
0 → 1 → 2 → 3 → 4 → 5 → 6 (iterate 4–6 as needed) → 7 → 8 → 9.
Write report sections *as each phase finishes* — don't leave writing to end.
