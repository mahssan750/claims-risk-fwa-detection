DROP TABLE IF EXISTS provider_features;

CREATE TABLE provider_features AS
WITH provider_rollup AS (
    SELECT
        provider_id,
        COUNT(*) AS claim_count,
        COUNT(DISTINCT beneficiary_id) AS member_count,
        COUNT(DISTINCT attending_physician) AS physician_count,
        COUNT(DISTINCT diagnosis_family) AS diagnosis_family_count,
        SUM(reimbursement_amount) AS total_reimbursement,
        AVG(reimbursement_amount) AS average_reimbursement,
        MAX(reimbursement_amount) AS maximum_reimbursement,
        SUM(COALESCE(deductible_amount, 0)) AS total_deductible,
        AVG(CASE WHEN claim_type = 'inpatient' THEN 1.0 ELSE 0.0 END) AS inpatient_share,
        AVG(exact_duplicate_flag * 1.0) AS exact_duplicate_rate,
        AVG(repeated_member_service_flag * 1.0) AS repeated_member_service_rate,
        AVG(inconsistent_date_flag * 1.0) AS inconsistent_date_rate,
        AVG(coding_inconsistency_flag * 1.0) AS coding_inconsistency_rate,
        AVG(cost_outlier_flag * 1.0) AS cost_outlier_rate,
        AVG(unusual_service_pattern_flag * 1.0) AS unusual_service_pattern_rate,
        AVG(nonpositive_amount_flag * 1.0) AS nonpositive_amount_rate,
        AVG(missing_physician_flag * 1.0) AS missing_physician_rate,
        AVG(service_days) AS average_service_days,
        AVG(age_at_claim) AS average_member_age,
        AVG(chronic_condition_count) AS average_chronic_conditions,
        AVG(annual_ip_reimbursement) AS average_annual_ip_reimbursement,
        AVG(annual_op_reimbursement) AS average_annual_op_reimbursement
    FROM claim_rule_flags
    GROUP BY provider_id
),
member_frequency AS (
    SELECT provider_id, beneficiary_id, COUNT(*) AS member_claim_count
    FROM claim_rule_flags
    GROUP BY provider_id, beneficiary_id
),
high_frequency_members AS (
    SELECT
        provider_id,
        AVG(CASE WHEN member_claim_count >= 5 THEN 1.0 ELSE 0.0 END) AS high_frequency_member_rate
    FROM member_frequency
    GROUP BY provider_id
),
diagnosis_counts AS (
    SELECT provider_id, diagnosis_family, COUNT(*) AS family_claim_count
    FROM claim_rule_flags
    GROUP BY provider_id, diagnosis_family
),
diagnosis_concentration AS (
    SELECT provider_id, MAX(family_claim_count) * 1.0 / SUM(family_claim_count) AS top_diagnosis_share
    FROM diagnosis_counts
    GROUP BY provider_id
)
SELECT
    p.*,
    p.claim_count * 1.0 / p.member_count AS claims_per_member,
    p.total_reimbursement / p.member_count AS reimbursement_per_member,
    h.high_frequency_member_rate,
    d.top_diagnosis_share,
    l.potential_fraud,
    l.dataset_split
FROM provider_rollup AS p
JOIN high_frequency_members AS h USING (provider_id)
JOIN diagnosis_concentration AS d USING (provider_id)
JOIN provider_labels AS l USING (provider_id);

CREATE UNIQUE INDEX idx_provider_features_id ON provider_features(provider_id);

