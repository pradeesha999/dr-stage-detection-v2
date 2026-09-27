# Where every figure goes in the report

All files live in `dr_project/outputs/figures/`. The report markdown
(`report/report.md`) already has a marked spot for each one — search the report for the
**figure number** and drop the image there, then keep the italic caption underneath.

**Priority column:** `CORE` = keep, it earns rubric marks. `TRIM` = drop first if you are over
the 20-page limit. Dropping all four TRIM figures saves roughly 2 pages.

| Fig | File | Report section | Suggested width | Priority |
|---|---|---|---|---|
| 1 | `01_class_distribution.png` | §2.2 Class distribution | full width | CORE |
| 2 | `01_samples_per_class.png` | §2.3 Image characteristics | ~60 % width (it is tall) | CORE |
| 3 | `01_image_statistics.png` | §2.3 Image characteristics | full width | TRIM |
| 4 | `01_left_right_agreement.png` | §2.4 Patient-level structure | full width | CORE |
| 5 | `01_split_distribution.png` | §2.4 Patient-level structure | half width | TRIM |
| 6 | `02_pipeline_steps.png` | §3.1 Pipeline | full width (tall — shrink to ~½ page) | CORE |
| 7 | `02_zoom_edge_enhancement.png` | §3.2 Justification of each step | full width | CORE |
| 8 | `02_clahe_vs_graham.png` | §3.3 Ben Graham comparison | ~50 % width | CORE |
| 9 | `02_quality_metrics.png` | §3.4 Quantitative evidence | full width | CORE |
| 10 | `02_histogram_before_after.png` | §3.4 Quantitative evidence | full width | TRIM |
| 11 | `03_augmentations_individual.png` | §4.1 Augmentation strategy | full width | CORE |
| 12 | `03_augmentation_stack.png` | §4.1 Augmentation strategy | full width | CORE |
| 13 | `03_sampling_strategies.png` | §4.2 Handling the imbalance | full width | CORE |
| 14 | `04_experiment_summary.png` | §5.5 Results of the ladder | full width | CORE |
| 15 | `04_final_curves.png` | §5.6 Training curves | full width | CORE |
| 16 | `04_final_qwk_f1.png` | §5.6 Training curves | half width | CORE |
| 17 | `04_ablation_curves.png` | §5.6 Training curves | **appendix only** | TRIM |
| 18 | `05_confusion_matrix.png` | §6.2 Per-class performance | full width | CORE |
| 19 | `05_error_distribution.png` | §6.3 Error analysis | full width | CORE |
| 20 | `05_worst_errors.png` | §6.3 Error analysis | full width | CORE |
| 21 | `05_roc_pr_curves.png` | §6.4 ROC and PR curves | full width | CORE |
| 22 | `05_referable_dr.png` | §6.5 Referable-DR performance | half width | CORE |
| 23 | `05_gradcam_per_class.png` | §6.6 Explainability | ~60 % width (tall) | CORE |
| 24 | `05_gradcam_errors.png` | §6.6 Explainability | full width | CORE |

## Two images you still have to make

| Placeholder in report | Section | How to produce it |
|---|---|---|
| `_[FIGURE: architecture diagram …]_` | §5.2 | Draw the block diagram in PowerPoint or draw.io from the code block directly above the placeholder (Input → Rescaling → ResNet50-V2 → GAP → Dropout → Dense256 → Dropout → Dense5). Slide 14 of the presentation already contains this diagram — screenshot it. |
| `_[FIGURE: screenshot of the running app …]_` | §8 | Run `python app/app.py`, upload a Moderate or Proliferative test image, screenshot the whole window showing prediction + probabilities + Grad-CAM. |

## One thing you must fill in

`_[INSERT URL HERE]_` in §8 — the hosted video link. Also add it to the title block at the top
of the report (`**Demo video:** _[URL to be added]_`).

## Page-budget note

The report is ~5,800 words. With all 24 figures at the widths above it lands around 19–20 pages.
If you overrun: drop the five TRIM figures first, then shrink Figures 2, 6 and 23 (the tall
multi-row grids) — they stay readable at half height.
