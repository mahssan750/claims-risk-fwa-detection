import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from claims_risk.config import PATHS, RANDOM_SEED
from claims_risk.prepare import provider_features

IDENTIFIER_COLUMNS = ["provider_id", "potential_fraud", "dataset_split"]
RULE_FEATURES = [
    "exact_duplicate_rate",
    "repeated_member_service_rate",
    "inconsistent_date_rate",
    "coding_inconsistency_rate",
    "cost_outlier_rate",
    "unusual_service_pattern_rate",
    "nonpositive_amount_rate",
    "missing_physician_rate",
]


@dataclass(frozen=True)
class DataSplit:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    test_features: pd.DataFrame
    train_labels: pd.Series
    validation_labels: pd.Series
    test_labels: pd.Series


@dataclass(frozen=True)
class ModelOutputs:
    metrics: dict[str, object]
    comparison: pd.DataFrame
    risk_register: pd.DataFrame
    feature_importance: pd.DataFrame
    test_labels: pd.Series
    test_scores: np.ndarray
    test_predictions: np.ndarray


@dataclass(frozen=True)
class AnomalyOutputs:
    model: Pipeline
    scaler: MinMaxScaler
    split_scores: dict[str, np.ndarray]


@dataclass(frozen=True)
class RiskScoringComponents:
    supervised_model: Pipeline
    anomaly_model: Pipeline
    anomaly_scaler: MinMaxScaler
    feature_importance: pd.DataFrame
    feature_medians: pd.Series
    feature_scales: pd.Series
    threshold: float


def split_provider_data(provider_frame: pd.DataFrame) -> DataSplit:
    features = provider_frame.drop(columns=IDENTIFIER_COLUMNS)
    labels = provider_frame["potential_fraud"]
    train_mask = provider_frame["dataset_split"].eq("train")
    validation_mask = provider_frame["dataset_split"].eq("validation")
    test_mask = provider_frame["dataset_split"].eq("test")
    return DataSplit(
        features.loc[train_mask],
        features.loc[validation_mask],
        features.loc[test_mask],
        labels.loc[train_mask],
        labels.loc[validation_mask],
        labels.loc[test_mask],
    )


def supervised_candidates() -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=RANDOM_SEED)),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    return {"Logistic regression": logistic, "Random forest": forest}


def candidate_comparison(split: DataSplit) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    fitted_models: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    for name, candidate in supervised_candidates().items():
        candidate.fit(split.train_features, split.train_labels)
        probabilities = candidate.predict_proba(split.validation_features)[:, 1]
        fitted_models[name] = candidate
        # PR-AUC is the selection metric because potential-fraud providers are the minority class.
        rows.append(
            {
                "model": name,
                "validation_pr_auc": average_precision_score(split.validation_labels, probabilities),
                "validation_roc_auc": roc_auc_score(split.validation_labels, probabilities),
            }
        )
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False), fitted_models


def anomaly_scores(split: DataSplit) -> AnomalyOutputs:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("detector", IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_SEED, n_jobs=-1)),
        ]
    )
    train_raw = -model.fit(split.train_features).decision_function(split.train_features)
    scaler = MinMaxScaler().fit(train_raw.reshape(-1, 1))
    scores: dict[str, np.ndarray] = {}
    for name, feature_frame in [("validation", split.validation_features), ("test", split.test_features)]:
        raw_scores = -model.decision_function(feature_frame)
        scores[name] = np.clip(scaler.transform(raw_scores.reshape(-1, 1)).ravel(), 0, 1)
    return AnomalyOutputs(model, scaler, scores)


def rule_scores(feature_frame: pd.DataFrame) -> np.ndarray:
    weighted_rate = feature_frame[RULE_FEATURES].sum(axis=1).to_numpy()
    return 1 - np.exp(-4 * weighted_rate)


def ensemble_scores(supervised: np.ndarray, anomaly: np.ndarray, rules: np.ndarray) -> np.ndarray:
    # Fixed weights keep the dominant predictive signal while retaining independent rule and anomaly evidence.
    return 0.65 * supervised + 0.20 * anomaly + 0.15 * rules


def f2_threshold(labels: pd.Series, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = 4 * precision[:-1] + recall[:-1]
    # F2 favors recall so fewer risky providers are missed at the investigation stage.
    f2_values = np.divide(
        5 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(thresholds), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(f2_values))])


def classification_metrics(labels: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions)
    flagged_providers = int(predictions.sum())
    return {
        "threshold": threshold,
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "f2": fbeta_score(labels, predictions, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "flagged_providers": flagged_providers,
        "review_rate": flagged_providers / len(predictions),
    }


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = np.abs(classifier.coef_[0])
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)


def rule_reasons(provider_row: pd.Series) -> list[str]:
    labels = {
        "exact_duplicate_rate": "duplicate claim pattern",
        "repeated_member_service_rate": "repeated member-service activity",
        "coding_inconsistency_rate": "coding inconsistency",
        "inconsistent_date_rate": "inconsistent service dates",
        "cost_outlier_rate": "cost outlier",
        "unusual_service_pattern_rate": "unusual service mix",
        "nonpositive_amount_rate": "non-positive reimbursement",
        "missing_physician_rate": "missing physician detail",
    }
    return [label for feature, label in labels.items() if provider_row[feature] > 0]


def feature_reasons(
    provider_row: pd.Series,
    importance_frame: pd.DataFrame,
    medians: pd.Series,
    robust_scales: pd.Series,
) -> list[str]:
    top_features = importance_frame.head(10)["feature"]
    scaled_deviation = (provider_row[top_features] - medians[top_features]).abs() / robust_scales[top_features]
    importance = importance_frame.set_index("feature").loc[top_features, "importance"]
    deviations = scaled_deviation * importance
    return [feature.replace("_", " ") for feature in deviations.nlargest(3).index]


def provider_reason(
    provider_row: pd.Series,
    importance_frame: pd.DataFrame,
    medians: pd.Series,
    robust_scales: pd.Series,
) -> str:
    reasons = rule_reasons(provider_row)
    reasons.extend(feature_reasons(provider_row, importance_frame, medians, robust_scales))
    return "; ".join(dict.fromkeys(reasons))


def final_risk_register(
    provider_frame: pd.DataFrame,
    components: RiskScoringComponents,
) -> pd.DataFrame:
    features = provider_frame.drop(columns=IDENTIFIER_COLUMNS)
    supervised = components.supervised_model.predict_proba(features)[:, 1]
    anomaly_raw = -components.anomaly_model.decision_function(features)
    anomaly = np.clip(components.anomaly_scaler.transform(anomaly_raw.reshape(-1, 1)).ravel(), 0, 1)
    combined = ensemble_scores(supervised, anomaly, rule_scores(features))
    register = provider_frame[["provider_id", "potential_fraud", "dataset_split"]].copy()
    register["supervised_probability"] = supervised
    register["anomaly_score"] = anomaly
    register["rule_score"] = rule_scores(features)
    register["risk_score"] = combined
    threshold = components.threshold
    register["risk_tier"] = np.where(
        combined >= threshold, "High", np.where(combined >= threshold * 0.65, "Medium", "Low")
    )
    register["review_reasons"] = features.apply(
        provider_reason,
        axis=1,
        args=(components.feature_importance, components.feature_medians, components.feature_scales),
    )
    return register.sort_values("risk_score", ascending=False)


def save_model_artifacts(model: Pipeline, anomaly_model: Pipeline, anomaly_scaler: MinMaxScaler) -> None:
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    PATHS.artifacts.joinpath("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATHS.artifacts / "models" / "supervised_model.joblib")
    joblib.dump(anomaly_model, PATHS.artifacts / "models" / "isolation_forest.joblib")
    joblib.dump(anomaly_scaler, PATHS.artifacts / "models" / "anomaly_scaler.joblib")


def risk_scoring_components(
    model: Pipeline,
    anomaly: AnomalyOutputs,
    train_features: pd.DataFrame,
    threshold: float,
) -> RiskScoringComponents:
    importance_frame = feature_importance(model, train_features.columns.tolist())
    # Training-only IQR scaling prevents large-unit fields and held-out providers from shaping explanations.
    feature_medians = train_features.median()
    feature_scales = (train_features.quantile(0.75) - train_features.quantile(0.25)).replace(0, 1)
    return RiskScoringComponents(
        supervised_model=model,
        anomaly_model=anomaly.model,
        anomaly_scaler=anomaly.scaler,
        feature_importance=importance_frame,
        feature_medians=feature_medians,
        feature_scales=feature_scales,
        threshold=threshold,
    )


def metrics_with_run_context(
    metrics: dict[str, object],
    provider_frame: pd.DataFrame,
    selected_model: str,
) -> dict[str, object]:
    return metrics | {
        "providers": len(provider_frame),
        "positive_providers": int(provider_frame["potential_fraud"].sum()),
        "selected_model": selected_model,
    }


def train_and_evaluate() -> ModelOutputs:
    provider_frame = provider_features()
    split = split_provider_data(provider_frame)
    comparison, fitted_models = candidate_comparison(split)
    model = fitted_models[comparison.iloc[0]["model"]]
    anomaly = anomaly_scores(split)
    validation_scores = ensemble_scores(
        model.predict_proba(split.validation_features)[:, 1],
        anomaly.split_scores["validation"],
        rule_scores(split.validation_features),
    )
    threshold = f2_threshold(split.validation_labels, validation_scores)
    test_scores = ensemble_scores(
        model.predict_proba(split.test_features)[:, 1], anomaly.split_scores["test"], rule_scores(split.test_features)
    )
    metrics = metrics_with_run_context(
        classification_metrics(split.test_labels, test_scores, threshold),
        provider_frame,
        comparison.iloc[0]["model"],
    )
    components = risk_scoring_components(model, anomaly, split.train_features, threshold)
    register = final_risk_register(provider_frame, components)
    predictions = (test_scores >= threshold).astype(int)
    save_model_artifacts(model, anomaly.model, anomaly.scaler)
    return ModelOutputs(
        metrics=metrics,
        comparison=comparison,
        risk_register=register,
        feature_importance=components.feature_importance,
        test_labels=split.test_labels,
        test_scores=test_scores,
        test_predictions=predictions,
    )


def write_model_outputs(outputs: ModelOutputs) -> None:
    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    (PATHS.artifacts / "metrics.json").write_text(json.dumps(outputs.metrics, indent=2), encoding="utf-8")
    outputs.comparison.to_csv(PATHS.artifacts / "model_comparison.csv", index=False)
    outputs.feature_importance.to_csv(PATHS.artifacts / "feature_importance.csv", index=False)
    outputs.risk_register.head(500).to_csv(PATHS.artifacts / "provider_risk_register.csv", index=False)
