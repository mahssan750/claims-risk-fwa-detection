# Methodology

## Decision question

Which providers should a claims-risk analyst review first, and what evidence supports that priority?

The system returns an investigation queue. It does not determine intent, fraud, payment eligibility or medical necessity.

## Analytical units

- **Claim level:** rule checks for duplicate, coding, date, cost and service-pattern signals.
- **Provider level:** supervised and unsupervised models use aggregated volume, utilization, reimbursement, member-mix and claim-rule features.
- **Target:** the published provider-level `PotentialFraud` label.

## Data preparation

Python validates the four downloaded files by MD5, assigns provider-level train/validation/test partitions, harmonizes inpatient and outpatient claims, counts populated diagnosis and procedure fields, derives service duration and member age, and aggregates chronic-condition counts. Sensitive beneficiary identifiers remain local and are not exported in published artifacts.

## SQL analytics

SQLite implements the claim rules and provider aggregation. Peer cost reference statistics are learned from training providers only, within claim type and three-character diagnosis family when the peer group has at least 30 claims. Provider features include claim/member counts, reimbursement totals, inpatient share, diagnosis concentration and rule prevalence.

## Modeling and leakage control

Providers are stratified into 60% training, 20% validation and 20% test partitions. All claims for a provider therefore remain in one analytical partition after aggregation.

1. Logistic regression and random forest are fitted on the training set.
2. The candidate with the highest validation PR-AUC is selected.
3. Isolation Forest is fitted without labels on training features.
4. Rule, supervised and anomaly scores are combined with fixed weights of 15%, 65% and 20%.
5. The F2-optimal threshold is chosen on validation data.
6. Final metrics are computed once on the untouched test set.

F2 weights recall more heavily than precision because the intended use is investigation prioritization.

## Explainability

Every provider receives the separate supervised, anomaly and rule scores. Review reasons combine triggered rule families with the provider's largest weighted deviations among globally important features. These reasons are directional investigation cues, not causal explanations.

## Limitations

- The published label is potential rather than adjudicated fraud.
- The dataset is historical and may not reflect current coding, reimbursement or utilization patterns.
- Provider-level labels cannot identify which individual claim is fraudulent.
- Rare services and cost outliers can reflect legitimate case mix or specialization.
- Performance estimates do not transfer automatically to another payer or market.
- Production use requires temporal validation, calibration, drift monitoring, governance and human review.

