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
* **Genuine acceptance/rejection** – counts of correct accept / correct reject decisions.
* **Accuracy, precision, recall, F1** – on the final 4-way decision (member_1/2/3 or *unknown*=rejected); macro-averaged F1 given.
* 95 % Wilson confidence intervals are reported because n is 9–18 per group.

## Results

### 1. Dataset Inspection and Augmentation
The dataset contained 9 original facial images from 3 enrolled members. No missing, corrupt, or duplicate images were identified.

Four augmentation techniques were applied: rotation, horizontal flipping, grayscale conversion, and brightness adjustment. This generated 36 additional images, bringing the total to 45 image records.

### 2. Failure Handling
Seven unit tests were conducted using a stand-in embedder to test different failure scenarios, including missing files, unreadable images, face detection failures, and embedding errors. All seven tests passed.

### 3. Facial Verification Results
The system used FaceNet512 for feature extraction, RetinaFace for face detection, and a cosine similarity threshold of 0.70.

The initial evaluation produced:
- Closed-set accuracy: **100%** (9/9).
- False Rejection Rate (FRR): **0%**.
- False Acceptance Rate (FAR): **0%** across 18 enrolled-impostor trials.
- Four-class accuracy and macro-F1: **100%**.

### 4. Unknown-Person Testing
The system was tested with 9 images from 3 people who were not enrolled in the system. All 9 images were processed successfully and correctly rejected.

- Correct rejections: 9
- False acceptances: 0
- Unknown-person FAR: **0%**
- Highest similarity score: 0.626, below the 0.70 threshold.

These images were tested separately and were not added to the enrolled dataset or templates.

### 5. Combined Evaluation
The combined binary evaluation recorded 100% accuracy, precision, recall and F1-score across 27 samples. Both FAR and FRR were 0%.

### 6. Summary
The system correctly identified enrolled users and rejected unknown individuals across the available test samples. However, the dataset was small, with only 3 enrolled members and 3 unknown individuals. Therefore, these results do not establish that the system is reliable for real-world deployment. Further testing with more individuals and images captured under different conditions is required.

Loss was not reported because the system uses a pretrained face-embedding model and threshold-based verification rather than training a new classifier.

## Limitations
Only 3 identities × 3 originals; single session each; capture conditions differ per member; unknown-person tests in the leave-one-member-out protocol use just two enrolled identities; no liveness; no demographic analysis. Perfect scores on this set would be weak evidence of real-world performance. Needed: multiple sessions per member, ≥ 20–50 consented unknown people, independent held-out genuine photos.
