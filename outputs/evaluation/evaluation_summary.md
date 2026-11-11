# Evaluation summary

Generated: 2026-11-11T11:26:48+00:00 UTC
Model: Facenet512 | detector: retinaface | threshold: 0.7
Image records: 45 (9 original and 36 augmented)

## Initial evaluation
- Closed-set top-1 accuracy: 1.000 (n=9)
- FRR: 0.000 (genuine n=9, 95% CI [0.00, 0.30])
- FAR for enrolled impostors: 0.000 (n=18, 95% CI [0.00, 0.18])
- Earlier unknown probes: FAR 0.000 (n=9, 95% CI [0.00, 0.30])
- Initial four-class accuracy / macro-F1: 1.000 / 1.000 (n=18)

## New unknown-person test
- Images supplied: 9
- Processed: 9 | failed: 0
- Correct rejections: 9 | false acceptances: 0
- Unknown-person FAR: 0.000
- Highest unknown similarity: 0.626

## Combined binary evaluation
- Samples: 27 | accuracy: 1.000
- Precision: 1.000 | recall: 1.000 | F1: 1.000
- FAR: 0.000 | FRR: 0.000

The unknown-person images were tested separately against the existing enrolled templates. They were not added to the enrolled dataset or used to update the templates.

Caution: the dataset includes only three enrolled members and three unknown people. The results describe this test set and do not establish real-world security.
Loss is not reported because the notebook uses a pretrained embedding model and threshold-based verification rather than training a new classifier.