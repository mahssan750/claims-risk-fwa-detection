# Executive summary

The workflow screened **558,211 claims** across **5,410 providers**. The published label set identifies **506 providers** as potential fraud cases.

On a held-out provider test set, the combined rules + supervised + anomaly ensemble achieved:

- Precision: **44.1%**
- Recall: **82.2%**
- F2 score: **0.701**
- PR-AUC: **0.659**
- ROC-AUC: **0.934**
- False positives: **105**
- False negatives: **18**

These outputs prioritize investigation; they do not prove fraud. Analysts should validate coding, medical necessity, contracts and documentation before taking action.
