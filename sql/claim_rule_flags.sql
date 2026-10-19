DROP TABLE IF EXISTS claim_rule_flags;

CREATE TABLE claim_rule_flags AS
WITH claim_windows AS (
    SELECT
        c.*,
        SUBSTR(COALESCE(c.diagnosis_code, 'MISSING'), 1, 3) AS diagnosis_family,
        COUNT(*) OVER (
            PARTITION BY c.beneficiary_id, c.provider_id, c.claim_start_date,
                c.claim_end_date, c.reimbursement_amount,
                COALESCE(c.diagnosis_code, ''), COALESCE(c.procedure_code, '')
        ) AS matching_claim_count,
        COUNT(*) OVER (
            PARTITION BY c.beneficiary_id, c.provider_id, c.claim_start_date
        ) AS same_day_provider_member_count,
        COUNT(*) OVER (
            PARTITION BY c.claim_type,
                SUBSTR(COALESCE(c.diagnosis_code, 'MISSING'), 1, 3),
                COALESCE(c.procedure_code, 'NONE')
        ) AS service_pattern_count
    FROM claims AS c
),
service_stats AS (
    SELECT
        claim_type,
        diagnosis_family,
        COUNT(*) AS service_claim_count,
        AVG(reimbursement_amount) AS mean_reimbursement,
        AVG(reimbursement_amount * reimbursement_amount) AS mean_square_reimbursement
    FROM claim_windows AS w
    JOIN provider_labels AS l
        ON w.provider_id = l.provider_id
        AND l.dataset_split = 'train'
    GROUP BY w.claim_type, w.diagnosis_family
)
SELECT
    w.*,
    CASE WHEN w.matching_claim_count > 1 THEN 1 ELSE 0 END AS exact_duplicate_flag,
    CASE WHEN w.same_day_provider_member_count > 2 THEN 1 ELSE 0 END AS repeated_member_service_flag,
    CASE
        WHEN w.claim_end_date < w.claim_start_date THEN 1
        WHEN w.claim_type = 'inpatient' AND w.discharge_date < w.admission_date THEN 1
        ELSE 0
    END AS inconsistent_date_flag,
    CASE
        WHEN w.diagnosis_count = 0 THEN 1
        WHEN w.procedure_count > 0 AND w.diagnosis_count = 0 THEN 1
        WHEN w.claim_type = 'inpatient'
            AND (w.admission_diagnosis_code IS NULL OR w.admission_date IS NULL OR w.discharge_date IS NULL)
            THEN 1
        ELSE 0
    END AS coding_inconsistency_flag,
    CASE
        WHEN s.service_claim_count >= 30
            AND w.reimbursement_amount > s.mean_reimbursement
            AND (w.reimbursement_amount - s.mean_reimbursement)
                * (w.reimbursement_amount - s.mean_reimbursement)
                > 9 * MAX(s.mean_square_reimbursement
                    - s.mean_reimbursement * s.mean_reimbursement, 0)
            THEN 1
        ELSE 0
    END AS cost_outlier_flag,
    CASE WHEN w.service_pattern_count <= 5 THEN 1 ELSE 0 END AS unusual_service_pattern_flag,
    CASE WHEN w.reimbursement_amount <= 0 THEN 1 ELSE 0 END AS nonpositive_amount_flag,
    CASE WHEN w.attending_physician IS NULL OR w.attending_physician = '' THEN 1 ELSE 0 END AS missing_physician_flag
FROM claim_windows AS w
LEFT JOIN service_stats AS s
    ON w.claim_type = s.claim_type
    AND w.diagnosis_family = s.diagnosis_family;

CREATE INDEX idx_flags_provider ON claim_rule_flags(provider_id);
CREATE INDEX idx_flags_member ON claim_rule_flags(beneficiary_id);

