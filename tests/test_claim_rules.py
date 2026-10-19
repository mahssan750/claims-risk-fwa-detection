import sqlite3
from pathlib import Path

import pandas as pd

RULE_SQL = Path(__file__).resolve().parents[1] / "sql" / "claim_rule_flags.sql"


def claim_row(claim_id: str, amount: float = 100.0) -> dict[str, object]:
    return {
        "beneficiary_id": "BENE1",
        "claim_id": claim_id,
        "provider_id": "PRV1",
        "claim_type": "outpatient",
        "claim_start_date": "2009-01-01",
        "claim_end_date": "2009-01-01",
        "admission_date": None,
        "discharge_date": None,
        "reimbursement_amount": amount,
        "deductible_amount": 0,
        "attending_physician": "PHY1",
        "operating_physician": None,
        "other_physician": None,
        "admission_diagnosis_code": None,
        "diagnosis_count": 1,
        "procedure_count": 0,
        "diagnosis_code": "2500",
        "procedure_code": None,
        "service_days": 1,
        "age_at_claim": 65,
        "chronic_condition_count": 2,
        "annual_ip_reimbursement": 0,
        "annual_op_reimbursement": 500,
    }


def evaluated_flags(claim_rows: list[dict[str, object]]) -> pd.DataFrame:
    with sqlite3.connect(":memory:") as connection:
        pd.DataFrame(claim_rows).to_sql("claims", connection, index=False)
        pd.DataFrame([{"provider_id": "PRV1", "potential_fraud": 0, "dataset_split": "train"}]).to_sql(
            "provider_labels", connection, index=False
        )
        connection.executescript(RULE_SQL.read_text(encoding="utf-8"))
        return pd.read_sql_query("SELECT * FROM claim_rule_flags ORDER BY claim_id", connection)


def test_repeated_identical_claims_generate_duplicate_and_same_day_flags() -> None:
    flags = evaluated_flags([claim_row("CLM1"), claim_row("CLM2"), claim_row("CLM3")])

    assert flags["exact_duplicate_flag"].tolist() == [1, 1, 1]
    assert flags["repeated_member_service_flag"].tolist() == [1, 1, 1]


def test_extreme_service_cost_is_flagged_against_peer_group() -> None:
    peer_claims = [claim_row(f"CLM{index:02d}") for index in range(30)]
    peer_claims.append(claim_row("CLM99", amount=10_000))

    flags = evaluated_flags(peer_claims)

    assert flags.loc[flags["claim_id"].eq("CLM99"), "cost_outlier_flag"].item() == 1
    assert flags.loc[flags["claim_id"].eq("CLM00"), "cost_outlier_flag"].item() == 0


def test_invalid_inpatient_dates_and_missing_codes_are_flagged() -> None:
    invalid_claim = claim_row("CLM1") | {
        "claim_type": "inpatient",
        "claim_start_date": "2009-02-02",
        "claim_end_date": "2009-02-01",
        "admission_date": "2009-02-02",
        "discharge_date": "2009-02-01",
        "diagnosis_count": 0,
        "diagnosis_code": None,
        "admission_diagnosis_code": None,
    }

    flags = evaluated_flags([invalid_claim])

    assert flags.loc[0, "inconsistent_date_flag"] == 1
    assert flags.loc[0, "coding_inconsistency_flag"] == 1
