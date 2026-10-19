# Executive summary

The workflow screened **558,211 claims** across **5,410 providers**. The published label set identifies **506 providers** as potential fraud cases.

On a held-out provider test set, the combined rules + supervised + anomaly ensemble achieved:

- Precision: **45.1%**
- Recall: **81.2%**
- F2 score: **0.700**
- PR-AUC: **0.663**
- ROC-AUC: **0.936**
- False positives: **100**
- False negatives: **19**
- Providers sent to review: **182 (16.8% of the test set)**

These outputs prioritize investigation; they do not prove fraud. Analysts should validate coding, medical necessity, contracts and documentation before taking action.
