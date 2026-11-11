# Model card – face verification prototype

| Field | Value |
|---|---|
| Name / version | `face_model` v0.1 (academic prototype) |
| Pretrained architecture | FaceNet, 512-d embeddings (`Facenet512` in DeepFace); detector RetinaFace by default (auto-fallback, see `face_config.json`) |
| Weights | Downloaded by DeepFace on first use (not bundled; not re-hosted). Obtain by running the notebook or any `DeepFace.represent` call with internet access. |
| Intended purpose | Coursework demonstration of the face stage (stage 1) of a multimodal authentication pipeline |
| Training data | None – no training/fine-tuning. Enrolment uses 3 originals (+ 12 augmented copies) per member = 9 originals, 36 augmented in total; unknown-person images are never used for enrolment |
| Input | Path to an image file (JPEG/PNG) containing one face |
| Output | `{"verified", "user_id", "score", "message"}` + diagnostics `status`, `best_match`, `threshold`, `num_faces` |
| Method | Cosine similarity to per-member templates; accept if ≥ 0.70, else reject |
| Evaluation | Protocol defined and implemented (`src/evaluation.py`, tested on synthetic vectors for logic only). **Measured results on the real embeddings are produced when the notebook is run in Colab** – see `outputs/evaluation/evaluation_summary.md`. No numbers are quoted here because none were measured in the build environment (no internet/TensorFlow). |

## Known limitations
* Tiny data: 3 people, 3 photos each, one session per person. Statistical confidence is low (wide intervals).
* Capture conditions (camera, background, clothing) differ between members, so identity is partly confounded with capture conditions.
* The initial unknown-person trials treat enrolled members as "strangers". Notebook Section 11 additionally tests 9 images of 3 never-enrolled people from `data/test_unknown/` (results are produced by running the notebook; 3 people is far too few for any security claim).
* No liveness/anti-spoofing: a printed photo or screen replay could be accepted.
* Threshold is the library default, not calibrated on our population.
* Uses the largest face if several are present.

## Bias and fairness
Face recognition error rates are known to vary across skin tones, genders, ages and imaging conditions, and public training sets are not demographically balanced. No subgroup analysis is possible with three people, so no fairness claim is made. Any real use would require evaluation on a diverse, consented population.

## Privacy
Facial images and embeddings are biometric personal data. Keep repositories private, share only with consenting team members, do not commit templates/embeddings publicly, and delete data when the project ends. The code processes images locally; only model weights are downloaded. Consent of members is assumed for academic use but has **not** been verified by this project – the team should confirm it.

## Appropriate / inappropriate use
Appropriate: classroom demonstration, integration testing of the face → voice → recommendation flow.
Inappropriate: real financial, access-control or identity decisions; any use on people who have not consented.
