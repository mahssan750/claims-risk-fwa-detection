import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from claims_risk.config import PATHS, RANDOM_SEED

DIAGNOSIS_COLUMNS = [f"ClmDiagnosisCode_{index}" for index in range(1, 11)]
PROCEDURE_COLUMNS = [f"ClmProcedureCode_{index}" for index in range(1, 7)]
CHRONIC_COLUMNS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


def source_path(file_name: str) -> Path:
    file_path = PATHS.raw / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {file_path}. Run the download command first.")
    return file_path


def code_values(code_series: pd.Series) -> pd.Series:
    return code_series.astype("string").str.replace(r"\.0$", "", regex=True)


def claim_columns(claim_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=claim_frame.index)
    normalized["diagnosis_count"] = claim_frame[DIAGNOSIS_COLUMNS].notna().sum(axis=1)
    normalized["procedure_count"] = claim_frame[PROCEDURE_COLUMNS].notna().sum(axis=1)
    normalized["diagnosis_code"] = code_values(claim_frame["ClmDiagnosisCode_1"])
    normalized["procedure_code"] = code_values(claim_frame["ClmProcedureCode_1"])
    return normalized


def standardize_claims(claim_frame: pd.DataFrame, claim_type: str) -> pd.DataFrame:
    codes = claim_columns(claim_frame)
    standardized = pd.DataFrame(
        {
            "beneficiary_id": claim_frame["BeneID"],
            "claim_id": claim_frame["ClaimID"],
            "provider_id": claim_frame["Provider"],
            "claim_type": claim_type,
            "claim_start_date": claim_frame["ClaimStartDt"],
            "claim_end_date": claim_frame["ClaimEndDt"],
            "admission_date": claim_frame.get("AdmissionDt"),
            "discharge_date": claim_frame.get("DischargeDt"),
            "reimbursement_amount": claim_frame["InscClaimAmtReimbursed"],
            "deductible_amount": claim_frame["DeductibleAmtPaid"],
            "attending_physician": claim_frame["AttendingPhysician"],
            "operating_physician": claim_frame["OperatingPhysician"],
            "other_physician": claim_frame["OtherPhysician"],
            "admission_diagnosis_code": code_values(claim_frame["ClmAdmitDiagnosisCode"]),
        }
    )
    return pd.concat([standardized, codes], axis=1)


def beneficiary_features(beneficiary_frame: pd.DataFrame) -> pd.DataFrame:
    chronic_count = beneficiary_frame[CHRONIC_COLUMNS].eq(1).sum(axis=1)
    return pd.DataFrame(
        {
            "beneficiary_id": beneficiary_frame["BeneID"],
            "date_of_birth": beneficiary_frame["DOB"],
            "chronic_condition_count": chronic_count,
            "annual_ip_reimbursement": beneficiary_frame["IPAnnualReimbursementAmt"],
            "annual_op_reimbursement": beneficiary_frame["OPAnnualReimbursementAmt"],
        }
    )


def add_claim_derived_fields(claims: pd.DataFrame) -> pd.DataFrame:
    start_dates = pd.to_datetime(claims["claim_start_date"])
    end_dates = pd.to_datetime(claims["claim_end_date"])
    birth_dates = pd.to_datetime(claims["date_of_birth"])
    enriched = claims.copy()
    enriched["service_days"] = (end_dates - start_dates).dt.days + 1
    enriched["age_at_claim"] = ((start_dates - birth_dates).dt.days / 365.25).round(1)
    return enriched.drop(columns="date_of_birth")


def load_claims() -> pd.DataFrame:
    inpatient = pd.read_csv(source_path("Train_Inpatientdata-1542865627584.csv"))
    outpatient = pd.read_csv(source_path("Train_Outpatientdata-1542865627584.csv"))
    claims = pd.concat(
        [standardize_claims(inpatient, "inpatient"), standardize_claims(outpatient, "outpatient")],
        ignore_index=True,
    )
    beneficiaries = pd.read_csv(source_path("Train_Beneficiarydata-1542865627584.csv"))
    return add_claim_derived_fields(claims.merge(beneficiary_features(beneficiaries), on="beneficiary_id"))


def load_provider_labels() -> pd.DataFrame:
    labels = pd.read_csv(source_path("Train-1542865627584.csv"))
    labels.columns = ["provider_id", "potential_fraud"]
    labels["potential_fraud"] = labels["potential_fraud"].eq("Yes").astype(int)
    return assign_provider_splits(labels)


def assign_provider_splits(labels: pd.DataFrame) -> pd.DataFrame:
    # Split first so SQL peer benchmarks cannot learn from validation or test providers.
    train_labels, remainder = train_test_split(
        labels,
        test_size=0.4,
        stratify=labels["potential_fraud"],
        random_state=RANDOM_SEED,
    )
    validation_labels, test_labels = train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder["potential_fraud"],
        random_state=RANDOM_SEED,
    )
    train_labels = train_labels.assign(dataset_split="train")
    validation_labels = validation_labels.assign(dataset_split="validation")
    test_labels = test_labels.assign(dataset_split="test")
    return pd.concat([train_labels, validation_labels, test_labels], ignore_index=True)


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE INDEX idx_claims_provider ON claims(provider_id)")
    connection.execute("CREATE INDEX idx_claims_member ON claims(beneficiary_id)")
    connection.execute("CREATE INDEX idx_claims_service ON claims(claim_type, diagnosis_code)")


def write_source_tables(connection: sqlite3.Connection) -> None:
    load_claims().to_sql("claims", connection, if_exists="replace", index=False, chunksize=10_000)
    load_provider_labels().to_sql("provider_labels", connection, if_exists="replace", index=False)
    create_indexes(connection)


def execute_sql_file(connection: sqlite3.Connection, sql_file: Path) -> None:
    connection.executescript(sql_file.read_text(encoding="utf-8"))


def build_analytics_database() -> Path:
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    if PATHS.database.exists():
        PATHS.database.unlink()
    with sqlite3.connect(PATHS.database) as connection:
        write_source_tables(connection)
        execute_sql_file(connection, PATHS.root / "sql" / "claim_rule_flags.sql")
        execute_sql_file(connection, PATHS.root / "sql" / "provider_features.sql")
    return PATHS.database


def provider_features() -> pd.DataFrame:
    with sqlite3.connect(PATHS.database) as connection:
        return pd.read_sql_query("SELECT * FROM provider_features", connection)
