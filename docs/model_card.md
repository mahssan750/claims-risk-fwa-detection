# Model card

## Intended use

Prioritize providers for manual claims review and demonstrate an explainable FWA analytics workflow.

## Out-of-scope use

- Automatic payment denial or recovery
- Accusations of fraud or provider sanctions
- Clinical decision-making
- Deployment without local validation and governance

## Model design

- **Supervised component:** random forest selected over logistic regression by validation PR-AUC
- **Unsupervised component:** Isolation Forest
- **Rules component:** eight claim-level indicators aggregated to provider level
- **Ensemble:** 65% supervised, 20% anomaly, 15% rules
- **Threshold:** validation F2 optimum

## Held-out performance

| Metric | Value |
|---|---:|
| Precision | 0.441 |
| Recall | 0.822 |
| F1 | 0.576 |
| F2 | 0.701 |
| PR-AUC | 0.659 |
| ROC-AUC | 0.934 |
| True negatives | 876 |
| False positives | 105 |
| False negatives | 18 |
| True positives | 83 |

## Monitoring recommendations

- Track alert rate, precision at analyst capacity and recall on adjudicated cases.
- Monitor feature and score drift by provider specialty and claim type.
- Review false positives for legitimate specialization or complex case mix.
- Recalibrate thresholds when prevalence, payment policy or review capacity changes.
- Audit performance across relevant groups before production use.

