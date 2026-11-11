# Evaluation report – face verification (Task 2)

## Status of results (read first)
| Item | Status |
|---|---|
| Dataset inspection (9 originals, no missing/corrupt/duplicate) | **Measured** – `outputs/evaluation/dataset_inspection.csv` |
| Augmentation (36 files) | **Measured** – `data/processed/augmentation_manifest.csv` |
| Failure handling of `verify_face` (missing file, unreadable file, no-face, below-threshold, embedder crash) | **Tested** with a stand-in embedder: 7 logic tests pass (`python -m unittest src.test_face_model`) |
| Evaluation code path | Smoke-run on *synthetic* vectors to check it executes and respects partitions. Synthetic numbers are meaningless and are **not reported**. |
| Face detection, embeddings, `image_features.csv`, templates | **Not generated in the build environment** (no internet, TensorFlow or pretrained weights available). Produced by running the notebook in Colab. |
| Real accuracy / precision / recall / F1 / FAR / FRR / confusion matrix | **Pending that notebook run.** They are written to `outputs/evaluation/evaluation_metrics.json` and `evaluation_summary.md`. Paste them here after running. |
| Unknown-person rejection with real strangers | **Images supplied and organised** (9 images, 3 people, `data/test_unknown/`); code unit-tested with a stand-in embedder (`src/test_unknown_testing.py`). **Results pending the notebook run** (Section 11) - none are reported here. |

## Methodology
Fixed threshold 0.70 cosine similarity (not tuned on this data). Three leakage-safe protocols:

1. **Leave-one-expression-out.** For each expression *e*: templates from the other two expressions (originals + their augmentations); probes = the held-out *original* of every member → 9 genuine trials; each probe against the other members' templates → 18 enrolled-impostor trials.
2. **Leave-one-member-out unknown trials.** For each member *u*: enrol only the other two; probe with *u*'s 3 originals → 9 trials that should all be rejected (real photos of people never enrolled in that run).
3. **Augmented-probe robustness** (separate, not independent): augmented copies of the held-out original.

Rules: an original and its augmentations are never split across enrolment/test; no probe contributes to the template that scores it; final deployed templates use all 45 enrolled-member image records (9 originals + 36 augmented copies) and therefore cannot be tested on independent images.

4. **Real unknown-person test (notebook Section 11).** 9 images of 3 never-enrolled people are scored against the final saved templates with the same pipeline and the fixed 0.70 threshold. Best score >= 0.70 = false acceptance; < 0.70 = correct rejection; detection/embedding failures are reported separately and excluded from rates. Templates and the enrolled feature table are hash-checked before/after to prove no leakage. Reported: unknown-only (correct rejections, false acceptances, FAR with 95% CI, accuracy) and a *combined* binary evaluation (initial 9 genuine + 9 leave-one-member-out probes + processed unknown images; FRR/recall carried over because no new genuine attempts are added) plus a combined four-class confusion matrix. Precision/recall/F1/FRR are undefined for the unknown-only set (no genuine attempts).

## Metric meanings
* **FRR** – genuine attempts rejected. **FAR** – impostor/unknown attempts accepted (separately for enrolled impostors and unknown persons).
* **Genuine acceptance / rejection** – counts of correct accept / correct reject decisions.
* **Accuracy, precision, recall, F1** – on the final 4-way decision (member_1/2/3 or *unknown*=rejected); macro-averaged F1 given.
* 95 % Wilson confidence intervals are reported because n is 9–18 per group.

## Results
*To be filled from `outputs/evaluation/evaluation_summary.md` after the Colab run (figures `05`–`09` in `outputs/figures/`).* Figures already generated: `01_original_images.png`, `02_dataset_distribution.png`, `04_augmentation_examples.png`. Figures `03` and `05`–`09` need the face model and are created by the notebook.

## Limitations
Only 3 identities × 3 originals; single session each; capture conditions differ per member; unknown-person tests in the leave-one-member-out protocol use just two enrolled identities; no liveness; no demographic analysis. Perfect scores on this set would be weak evidence of real-world performance. Needed: multiple sessions per member, ≥ 20–50 consented unknown people, independent held-out genuine photos.
