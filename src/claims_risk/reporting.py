import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from claims_risk.config import PATHS
from claims_risk.modeling import RULE_FEATURES, ModelOutputs

RULE_FLAGS = [feature.replace("_rate", "_flag") for feature in RULE_FEATURES]


def chart_path(file_name: str) -> Path:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    return PATHS.figures / file_name


def save_figure(file_name: str) -> None:
    plt.tight_layout()
    plt.savefig(chart_path(file_name), dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_balance() -> None:
    with sqlite3.connect(PATHS.database) as connection:
        labels = pd.read_sql_query(
            "SELECT potential_fraud, COUNT(*) AS providers FROM provider_labels GROUP BY potential_fraud", connection
        )
    labels["label"] = labels["potential_fraud"].map({0: "Not flagged", 1: "Potential fraud"})
    sns.barplot(data=labels, x="label", y="providers", hue="label", legend=False, palette=["#2C7FB8", "#D95F0E"])
    plt.title("Provider label distribution")
    plt.xlabel("")
    save_figure("provider_class_balance.png")


def plot_precision_recall(outputs: ModelOutputs) -> None:
    precision, recall, _ = precision_recall_curve(outputs.test_labels, outputs.test_scores)
    plt.plot(recall, precision, color="#2C7FB8", linewidth=2)
    plt.axhline(outputs.test_labels.mean(), color="#666666", linestyle="--", label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Ensemble precision-recall curve")
    plt.legend()
    save_figure("precision_recall_curve.png")


def plot_confusion_matrix(outputs: ModelOutputs) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        outputs.test_labels,
        outputs.test_predictions,
        display_labels=["Not flagged", "Potential fraud"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Held-out provider classification")
    save_figure("confusion_matrix.png")


def plot_feature_importance(outputs: ModelOutputs) -> None:
    top_features = outputs.feature_importance.head(12).sort_values("importance").copy()
    top_features["feature"] = top_features["feature"].str.replace("_", " ")
    sns.barplot(data=top_features, x="importance", y="feature", hue="feature", legend=False, color="#2C7FB8")
    plt.title(f"Top drivers — {outputs.metrics['selected_model']}")
    plt.xlabel("Relative importance")
    plt.ylabel("")
    save_figure("feature_importance.png")


def plot_risk_distribution(outputs: ModelOutputs) -> None:
    chart_frame = outputs.risk_register.copy()
    chart_frame["published label"] = chart_frame["potential_fraud"].map({0: "Not flagged", 1: "Potential fraud"})
    sns.histplot(
        data=chart_frame,
        x="risk_score",
        hue="published label",
        bins=35,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.axvline(outputs.metrics["threshold"], color="#222222", linestyle="--", label="Review threshold")
    plt.title("Provider risk-score distribution")
    plt.xlabel("Combined risk score")
    save_figure("risk_score_distribution.png")


def rule_summary() -> pd.DataFrame:
    expressions = ", ".join(f"SUM({flag}) AS {flag}" for flag in RULE_FLAGS)
    with sqlite3.connect(PATHS.database) as connection:
        totals = pd.read_sql_query(f"SELECT COUNT(*) AS claims, {expressions} FROM claim_rule_flags", connection).iloc[
            0
        ]
    rows = [
        {"rule": flag, "flagged_claims": int(totals[flag]), "flag_rate": float(totals[flag] / totals["claims"])}
        for flag in RULE_FLAGS
    ]
    return pd.DataFrame(rows).sort_values("flagged_claims", ascending=False)


def plot_rule_prevalence(summary: pd.DataFrame) -> None:
    chart_frame = summary.assign(rule=summary["rule"].str.replace("_flag", "", regex=False).str.replace("_", " "))
    sns.barplot(
        data=chart_frame.sort_values("flag_rate"), x="flag_rate", y="rule", hue="rule", legend=False, color="#41AB5D"
    )
    plt.gca().xaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    plt.title("Claim-level rule prevalence")
    plt.xlabel("Share of claims flagged")
    plt.ylabel("")
    save_figure("rule_prevalence.png")


def write_executive_summary(outputs: ModelOutputs, summary: pd.DataFrame) -> None:
    metrics = outputs.metrics
    summary_text = f"""# Executive summary

The workflow screened **558,211 claims** across **{metrics["providers"]:,} providers**. The published label set identifies **{metrics["positive_providers"]:,} providers** as potential fraud cases.

On a held-out provider test set, the combined rules + supervised + anomaly ensemble achieved:

- Precision: **{metrics["precision"]:.1%}**
- Recall: **{metrics["recall"]:.1%}**
- F2 score: **{metrics["f2"]:.3f}**
- PR-AUC: **{metrics["pr_auc"]:.3f}**
- ROC-AUC: **{metrics["roc_auc"]:.3f}**
- False positives: **{metrics["false_positives"]}**
- False negatives: **{metrics["false_negatives"]}**

These outputs prioritize investigation; they do not prove fraud. Analysts should validate coding, medical necessity, contracts and documentation before taking action.
"""
    (PATHS.artifacts / "executive_summary.md").write_text(summary_text, encoding="utf-8")
    summary.to_csv(PATHS.artifacts / "claim_rule_summary.csv", index=False)


def create_reports(outputs: ModelOutputs) -> None:
    sns.set_theme(style="whitegrid")
    plot_class_balance()
    plot_precision_recall(outputs)
    plot_confusion_matrix(outputs)
    plot_feature_importance(outputs)
    plot_risk_distribution(outputs)
    summary = rule_summary()
    plot_rule_prevalence(summary)
    write_executive_summary(outputs, summary)
