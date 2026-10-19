# Data dictionary

## Published source tables

| File | Grain | Main content |
|---|---|---|
| `Train_Inpatientdata-1542865627584.csv` | Inpatient claim | Provider, member, service dates, diagnoses, procedures, physicians and payment |
| `Train_Outpatientdata-1542865627584.csv` | Outpatient claim | Provider, member, service dates, diagnoses, procedures, physicians and payment |
| `Train_Beneficiarydata-1542865627584.csv` | Beneficiary | Demographics, coverage, chronic-condition indicators and annual payments |
| `Train-1542865627584.csv` | Provider | `PotentialFraud` label |

## Provider feature groups

| Group | Example fields | Purpose |
|---|---|---|
| Volume | `claim_count`, `member_count`, `physician_count` | Measures provider scale and reach |
| Cost | `total_reimbursement`, `maximum_reimbursement`, `reimbursement_per_member` | Captures overall and per-member financial exposure |
| Utilization | `claims_per_member`, `high_frequency_member_rate`, `inpatient_share` | Identifies unusually intensive patterns |
| Service mix | `diagnosis_family_count`, `top_diagnosis_share`, `average_service_days` | Describes concentration and care profile |
| Claim rules | `exact_duplicate_rate`, `cost_outlier_rate`, `coding_inconsistency_rate` | Aggregates transparent review signals |
| Member context | `average_member_age`, `average_chronic_conditions` | Controls for broad population complexity |

Gender, race, state and county are intentionally excluded from model features.

## Published outputs

| Artifact | Description |
|---|---|
| `metrics.json` | Held-out model metrics and selected threshold |
| `model_comparison.csv` | Validation PR-AUC and ROC-AUC for candidate classifiers |
| `claim_rule_summary.csv` | Count and prevalence of each transparent rule |
| `feature_importance.csv` | Global feature importance for the selected supervised model |
| `provider_risk_register.csv` | Top 500 providers by combined score with investigation reasons |

