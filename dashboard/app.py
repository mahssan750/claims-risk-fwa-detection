import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    risk_register = pd.read_csv(ARTIFACTS / "provider_risk_register.csv")
    rule_summary = pd.read_csv(ARTIFACTS / "claim_rule_summary.csv")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    return risk_register, rule_summary, metrics


def metric_row(metrics: dict[str, object]) -> None:
    columns = st.columns(4)
    columns[0].metric("Providers screened", f"{metrics['providers']:,}")
    columns[1].metric("Precision", f"{metrics['precision']:.1%}")
    columns[2].metric("Recall", f"{metrics['recall']:.1%}")
    columns[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")


def main() -> None:
    st.set_page_config(page_title="Claims Risk Review", layout="wide")
    st.title("Claims Risk and FWA Investigation Workbench")
    st.caption("Decision support for analyst review — a risk flag is not a finding of fraud.")
    risk_register, rule_summary, metrics = load_outputs()
    metric_row(metrics)
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(risk_register, x="risk_score", color="risk_tier", nbins=35, title="Provider risk distribution"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(rule_summary, x="flag_rate", y="rule", orientation="h", title="Claim rule prevalence"),
        use_container_width=True,
    )
    tiers = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
    display_columns = [
        "provider_id",
        "risk_tier",
        "risk_score",
        "supervised_probability",
        "anomaly_score",
        "rule_score",
        "review_reasons",
    ]
    st.dataframe(
        risk_register.loc[risk_register["risk_tier"].isin(tiers), display_columns],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
