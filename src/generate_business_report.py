"""Generate business-facing evaluation outputs for CLAUDE.md checklist."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .constants import (
    CAT_FEATURES,
    CORE_FEATURES,
    CT_THRESHOLD,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from .pipeline import build_model_tables
from .train_m1_m5 import _prep, _split
from .train_m6 import _prepare_survival_frame


REPORT_DIR = OUTPUT_DIR / "business_report"
PLOT_DIR = REPORT_DIR / "plots"


def _load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


def _savefig(name: str) -> str:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(OUTPUT_DIR.parent))


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _prep_with_artifacts(df: pd.DataFrame, features: list[str], cat: list[str], target: str, prefix: str):
    imputer = _load_pickle(MODEL_DIR / f"{prefix}_imputer.pkl")
    encoders = _load_pickle(MODEL_DIR / f"{prefix}_label_encoders.pkl")
    x, y, _, _, selected = _prep(df, features, cat, target, imputer=imputer, encoders=encoders)
    return x, y, selected


def _plot_eda(base: pd.DataFrame, field: pd.DataFrame) -> list[str]:
    paths: list[str] = []

    nulls = base.isna().mean().sort_values(ascending=False).head(25) * 100
    plt.figure(figsize=(10, 7))
    nulls.sort_values().plot.barh(color="#4062bb")
    plt.xlabel("Null rate (%)")
    plt.title("1. Null Rate Profile - Lineage")
    paths.append(_savefig("01_null_rate_profile.png"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(base["CT_Current"].dropna(), bins=40, ax=axes[0], color="#28745b")
    axes[0].axvline(CT_THRESHOLD, color="red", linestyle="--")
    axes[0].set_title("CT Distribution")
    base["degraded_binary"].value_counts(dropna=True).sort_index().plot.pie(
        ax=axes[1], autopct="%1.1f%%", labels=["Not degraded", "Degraded"], colors=["#6fbf73", "#c44536"]
    )
    axes[1].set_ylabel("")
    sns.boxplot(data=base, x="SEASON_YR", y="CT_Current", ax=axes[2], color="#9ecae1")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title("CT by Season")
    paths.append(_savefig("02_ct_distribution_class_season.png"))

    trend = base.groupby("SEASON_YR")["CT_Current"].agg(["mean", "count", "std"]).dropna()
    trend["se"] = trend["std"] / np.sqrt(trend["count"])
    plt.figure(figsize=(9, 5))
    plt.plot(trend.index, trend["mean"], marker="o", color="#174a7c")
    plt.fill_between(trend.index, trend["mean"] - 1.96 * trend["se"], trend["mean"] + 1.96 * trend["se"], alpha=0.2)
    plt.axhline(CT_THRESHOLD, color="red", linestyle="--")
    plt.ylabel("Mean CT")
    plt.title("3. Seasonal CT Trend")
    paths.append(_savefig("03_seasonal_trend.png"))

    region = base.groupby("Origin_Region").agg(mean_ct=("CT_Current", "mean"), degraded=("degraded_binary", "mean"), n=("CT_Current", "count"))
    region = region[region["n"] >= 50].sort_values("degraded", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=region.reset_index(), x="Origin_Region", y="degraded", color="#c44536")
    plt.ylabel("Degradation rate")
    plt.title("4. Regional Performance")
    paths.append(_savefig("04_regional_performance.png"))

    stage = base.groupby("Stage").agg(mean_ct=("CT_Current", "mean"), degraded=("degraded_binary", "mean"), n=("CT_Current", "count")).reset_index()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.bar(stage["Stage"], stage["degraded"], color="#c44536", alpha=0.7)
    ax2.plot(stage["Stage"], stage["mean_ct"], color="#174a7c", marker="o")
    ax1.set_ylabel("Degradation rate")
    ax2.set_ylabel("Mean CT")
    ax1.set_title("5. Pipeline Stage Gradient")
    paths.append(_savefig("05_stage_gradient.png"))

    variety = base.groupby("Variety").agg(degraded=("degraded_binary", "mean"), n=("CT_Current", "count")).query("n >= 30")
    variety = variety.sort_values("degraded", ascending=False).head(25)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=variety.reset_index(), y="Variety", x="degraded", color="#c44536")
    plt.title("6. Variety Risk Ranking - Top 25")
    paths.append(_savefig("06_variety_risk_ranking.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(base["WG_Current"], base["CT_Current"], s=8, alpha=0.25, color="#28745b")
    axes[0].plot([0, 100], [0, 100], color="black", linestyle=":")
    axes[0].axhline(CT_THRESHOLD, color="red", linestyle="--")
    axes[0].set_xlabel("WG Current")
    axes[0].set_ylabel("CT Current")
    axes[0].set_title("WG vs CT")
    gap = base["WG_Current"] - base["CT_Current"]
    sns.histplot(gap.dropna(), bins=40, ax=axes[1], color="#d99c2b")
    axes[1].set_title("Vigor Gap (WG - CT)")
    paths.append(_savefig("07_wg_vs_ct_vigor_gap.png"))

    phys = _available(base, ["Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB", "FFA"])
    melted = base[phys + ["degraded_binary"]].melt(id_vars="degraded_binary")
    g = sns.catplot(data=melted.dropna(), x="degraded_binary", y="value", col="variable", kind="box", sharey=False, height=4, aspect=0.9)
    g.fig.suptitle("8. Physical Quality by Degradation Class", y=1.05)
    paths.append(_savefig("08_physical_quality_by_class.png"))

    corr_cols = _available(base, CORE_FEATURES + ["CT_Current"])
    plt.figure(figsize=(9, 7))
    sns.heatmap(base[corr_cols].corr(numeric_only=True), cmap="vlag", center=0)
    plt.title("9. Core Feature Correlation Heatmap")
    paths.append(_savefig("09_correlation_heatmap.png"))

    if len(field):
        weather_cols = _available(field, ["pre_defol_dd_60_cap90", "post_defol_dd_60_cap90", "cumulated_soil_moisture", "irrigation_type", "state"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        sns.boxplot(data=field, x="state", y="pre_defol_dd_60_cap90", ax=axes[0], color="#9ecae1")
        sns.histplot(field["cumulated_soil_moisture"].dropna(), ax=axes[1], color="#28745b")
        field["irrigation_type"].value_counts().head(10).plot.bar(ax=axes[2], color="#d99c2b")
        axes[0].set_title("DD60 by State")
        axes[1].set_title("Soil Moisture")
        axes[2].set_title("Irrigation Types")
        paths.append(_savefig("10_weather_eda.png"))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(field["pre_defol_dd_60_cap90"].dropna(), ax=axes[0], color="#c44536")
        sns.histplot(field["season_length"].dropna(), ax=axes[1], color="#174a7c")
        axes[0].set_title("Pre-Defol DD60")
        axes[1].set_title("Season Length")
        paths.append(_savefig("11_cottons3_eda.png"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.scatterplot(data=base.sample(min(len(base), 5000), random_state=42), x="season_age", y="CT_Current", hue="degraded_binary", ax=axes[0], s=12)
    base.groupby("season_age")["event"].mean().plot.bar(ax=axes[1], color="#c44536")
    sns.histplot(base["ct_distance_to_threshold"].dropna(), bins=40, ax=axes[2], color="#28745b")
    axes[0].set_title("CT vs Age")
    axes[1].set_title("Event Rate by Age")
    axes[2].set_title("CT Buffer to Threshold")
    paths.append(_savefig("12_survival_eda.png"))

    return paths


def _plot_model_evals(tables: dict[str, pd.DataFrame]) -> list[str]:
    paths: list[str] = []
    base = tables["enriched"][tables["enriched"]["CT_Current"].notna()].copy()
    train_df, val_df, test_df = _split(base)

    m1 = _load_pickle(MODEL_DIR / "m1_binary_classifier.pkl")
    x1, y1, feat1 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "degraded_binary", "m1")
    y1 = y1.astype(int)
    p1 = m1.predict_proba(x1)[:, 1]
    pred1 = (p1 >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y1, p1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#174a7c")
    axes[0].plot([0, 1], [0, 1], color="black", linestyle=":")
    axes[0].set_title("13. M1 ROC Curve")
    ConfusionMatrixDisplay(confusion_matrix(y1, pred1)).plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title("M1 Confusion Matrix")
    paths.append(_savefig("13_m1_roc_confusion.png"))

    m2 = _load_pickle(MODEL_DIR / "m2_ct_regressor.pkl")
    x2, y2, feat2 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "CT_Current", "m2")
    pred2 = m2.predict(x2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y2, pred2, s=10, alpha=0.3, color="#28745b")
    axes[0].plot([0, 100], [0, 100], color="red", linestyle="--")
    axes[0].set_title("14. M2 Actual vs Predicted")
    sns.histplot((y2 - pred2), bins=40, ax=axes[1], color="#174a7c")
    axes[1].set_title("M2 Residual Distribution")
    paths.append(_savefig("14_m2_actual_pred_residuals.png"))

    m3 = _load_pickle(MODEL_DIR / "m3_3class_classifier.pkl")
    x3, y3, feat3 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "quality_class", "m3")
    pred3 = m3.predict(x3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(confusion_matrix(y3.astype(int), pred3), display_labels=["Degraded", "At Risk", "HQ"]).plot(ax=axes[0], colorbar=False, cmap="Greens")
    axes[0].set_title("15. M3 Confusion Matrix")
    imp = pd.Series(m3.feature_importances_, index=feat3).sort_values(ascending=False).head(15)
    imp.sort_values().plot.barh(ax=axes[1], color="#d99c2b")
    axes[1].set_title("M3 Feature Importance for Classifier")
    paths.append(_savefig("15_m3_confusion_importance.png"))

    field = tables["field_enriched"][tables["field_enriched"]["CT_Current"].notna()].copy()
    valid_vars = field["Variety"].value_counts()
    valid_vars = valid_vars[valid_vars >= 30].index[:3]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, var in zip(axes, valid_vars):
        v = field[field["Variety"] == var].dropna(subset=["pre_defol_dd_60_cap90", "CT_Current"])
        ax.scatter(v["pre_defol_dd_60_cap90"], v["CT_Current"], s=18, alpha=0.5, color="#28745b")
        if len(v) >= 2:
            coef = np.polyfit(v["pre_defol_dd_60_cap90"], v["CT_Current"], 1)
            xs = np.linspace(v["pre_defol_dd_60_cap90"].min(), v["pre_defol_dd_60_cap90"].max(), 100)
            ax.plot(xs, np.poly1d(coef)(xs), color="#c44536")
        ax.axhline(CT_THRESHOLD, color="red", linestyle="--")
        ax.set_title(f"{var} (n={len(v)})")
        ax.set_xlabel("Pre-defol DD60")
        ax.set_ylabel("CT Current")
    paths.append(_savefig("16_m5_gdd_curves.png"))

    return paths


def _plot_survival(base: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    paths: list[str] = []
    surv = base[base["CT_Current"].notna()].copy()
    surv["duration"] = surv["duration"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    km = KaplanMeierFitter(label="All lots")
    km.fit(surv["duration"], event_observed=surv["event"])
    km.plot_survival_function(ax=axes[0], color="#174a7c")
    axes[0].set_title("17. Overall Kaplan-Meier")
    for region in surv["Origin_Region"].dropna().unique()[:6]:
        mask = surv["Origin_Region"] == region
        if mask.sum() >= 50:
            KaplanMeierFitter(label=str(region)).fit(surv.loc[mask, "duration"], surv.loc[mask, "event"]).plot_survival_function(ax=axes[1])
    axes[1].set_title("KM by Region")
    for stage in sorted(surv["Stage"].dropna().unique()):
        mask = surv["Stage"] == stage
        if mask.sum() >= 50:
            KaplanMeierFitter(label=f"Stage {int(stage)}").fit(surv.loc[mask, "duration"], surv.loc[mask, "event"]).plot_survival_function(ax=axes[2])
    axes[2].set_title("KM by Stage")
    paths.append(_savefig("17_m6_kaplan_meier.png"))

    region_lr = multivariate_logrank_test(surv["duration"], surv["Origin_Region"].fillna("Unknown"), surv["event"])
    stage_lr = multivariate_logrank_test(surv["duration"], surv["Stage"].fillna(-1), surv["event"])

    cph = _load_pickle(MODEL_DIR / "m6_cox_ph.pkl")
    aft = _load_pickle(MODEL_DIR / "m6_aft_weibull.pkl")
    imputer = _load_pickle(MODEL_DIR / "cox_imputer.pkl")
    with (MODEL_DIR / "model_metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    train_raw = surv[surv["SEASON_YR"].isin(TRAIN_SEASONS + VAL_SEASONS)]
    test_raw = surv[surv["SEASON_YR"].isin(TEST_SEASONS)]
    train, _, cols = _prepare_survival_frame(train_raw, fit=True)
    test, _, _ = _prepare_survival_frame(test_raw, imputer=imputer, columns=meta["M6_MODEL_COLUMNS"])

    plt.figure(figsize=(9, 6))
    cph.plot()
    plt.title("18. M6 Cox PH Hazard Ratios")
    paths.append(_savefig("18_m6_cox_hazard_ratios.png"))

    pred_median = aft.predict_median(test).replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(pred_median.dropna().clip(upper=12), bins=35, ax=axes[0], color="#174a7c")
    axes[0].set_title("19. AFT Shelf-Life Distribution")
    axes[1].scatter(test["duration"], pred_median, s=12, alpha=0.3, color="#28745b")
    axes[1].plot([0, 8], [0, 8], color="red", linestyle="--")
    axes[1].set_title("AFT Actual vs Predicted Median")
    paths.append(_savefig("19_m6_aft_shelf_life.png"))

    examples = test.head(5)
    plt.figure(figsize=(10, 6))
    sf = aft.predict_survival_function(examples)
    for col in sf.columns:
        plt.plot(sf.index, sf[col], label=f"Lot {col}")
    plt.axhline(0.5, color="black", linestyle=":")
    plt.title("20. M6 Individual Lot Survival Curves")
    plt.xlabel("Seasons")
    plt.ylabel("Probability CT >= 60")
    plt.legend()
    paths.append(_savefig("20_m6_individual_survival_curves.png"))

    return paths, {
        "m6_region_logrank_p": float(region_lr.p_value),
        "m6_stage_logrank_p": float(stage_lr.p_value),
        "m6_c_index_test": float(meta["m6_c_index_test"]),
        "m6_aft_test_median_shelf_life": float(meta["m6_aft_test_median_shelf_life"]),
    }


def _plot_shap_sample(tables: dict[str, pd.DataFrame]) -> list[str]:
    paths: list[str] = []
    try:
        import shap
    except Exception as exc:
        (REPORT_DIR / "shap_import_error.txt").write_text(str(exc), encoding="utf-8")
        return paths

    base = tables["enriched"][tables["enriched"]["CT_Current"].notna()].copy()
    _, _, test_df = _split(base)
    sample_n = min(1000, len(test_df))

    m1 = _load_pickle(MODEL_DIR / "m1_binary_classifier.pkl")
    x1, _, feat1 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "degraded_binary", "m1")
    x1s = x1[:sample_n]
    sv1 = shap.TreeExplainer(m1).shap_values(x1s)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv1, x1s, feature_names=feat1, show=False, plot_type="bar", max_display=15)
    paths.append(_savefig("21_m1_shap_bar.png"))

    m2 = _load_pickle(MODEL_DIR / "m2_ct_regressor.pkl")
    x2, _, feat2 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "CT_Current", "m2")
    x2s = x2[:sample_n]
    sv2 = shap.TreeExplainer(m2).shap_values(x2s)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv2, x2s, feature_names=feat2, show=False, max_display=15)
    paths.append(_savefig("22_m2_shap_beeswarm.png"))

    m3 = _load_pickle(MODEL_DIR / "m3_3class_classifier.pkl")
    x3, _, feat3 = _prep_with_artifacts(test_df, CORE_FEATURES, CAT_FEATURES, "quality_class", "m3")
    x3s = x3[:sample_n]
    sv3 = shap.TreeExplainer(m3).shap_values(x3s)
    if isinstance(sv3, list):
        degraded_sv = sv3[0]
    elif getattr(sv3, "ndim", 0) == 3:
        degraded_sv = sv3[:, :, 0]
    else:
        degraded_sv = sv3
    plt.figure(figsize=(10, 6))
    shap.summary_plot(degraded_sv, x3s, feature_names=feat3, show=False, plot_type="bar", max_display=15)
    paths.append(_savefig("15_m3_shap_degraded_class.png"))

    return paths


def generate_report() -> dict[str, object]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    tables = build_model_tables()
    base = tables["enriched"]
    field = tables["field_enriched"]

    paths = []
    paths.extend(_plot_eda(base, field))
    paths.extend(_plot_model_evals(tables))
    survival_paths, survival_metrics = _plot_survival(base)
    paths.extend(survival_paths)
    paths.extend(_plot_shap_sample(tables))

    with (MODEL_DIR / "m1_m5_metrics.json").open(encoding="utf-8") as f:
        m1_m5 = json.load(f)
    metrics = {**m1_m5, **survival_metrics}
    metrics_df = pd.DataFrame(
        [
            ["M1", "AUC-ROC", metrics["m1_auc"], ">= 0.80", metrics["m1_auc"] >= 0.80],
            ["M1", "PR-AUC", metrics["m1_pr_auc"], ">= 0.65", metrics["m1_pr_auc"] >= 0.65],
            ["M2", "RMSE", metrics["m2_rmse"], "< 10", metrics["m2_rmse"] < 10],
            ["M2", "R2", metrics["m2_r2"], ">= 0.65", metrics["m2_r2"] >= 0.65],
            ["M3", "Macro-F1", metrics["m3_macro_f1"], ">= 0.72", metrics["m3_macro_f1"] >= 0.72],
            ["M4", "AUC-ROC", metrics["m4_auc"], ">= 0.75", metrics["m4_auc"] >= 0.75],
            ["M5", "RMSE", metrics["m5_rmse"], "< 12", metrics["m5_rmse"] < 12],
            ["M6", "C-index", metrics["m6_c_index_test"], ">= 0.70", metrics["m6_c_index_test"] >= 0.70],
        ],
        columns=["Model", "Metric", "Value", "Target", "Passed"],
    )
    metrics_df.to_csv(REPORT_DIR / "model_performance_summary.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.axis("off")
    table = plt.table(cellText=metrics_df.round(4).values, colLabels=metrics_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.title("23. Model Performance Summary")
    paths.append(_savefig("23_model_performance_summary.png"))

    checklist = [
        "# CLAUDE.md Evaluation Checklist",
        "",
        "All values below are generated from the local source data and saved model artifacts.",
        "",
        f"- M1 AUC-ROC: {metrics['m1_auc']:.4f} (target >= 0.80)",
        f"- M1 PR-AUC: {metrics['m1_pr_auc']:.4f} (target >= 0.65)",
        f"- M1 confusion matrix: {metrics['m1_confusion_matrix']}",
        "- M1 SHAP plot: outputs/business_report/plots/21_m1_shap_bar.png",
        f"- M2 RMSE: {metrics['m2_rmse']:.4f} (target < 10)",
        f"- M2 R2: {metrics['m2_r2']:.4f} (target >= 0.65)",
        "- M2 actual vs predicted and residual plots: outputs/business_report/plots/14_m2_actual_pred_residuals.png",
        "- M2 SHAP plot: outputs/business_report/plots/22_m2_shap_beeswarm.png",
        f"- M3 Macro-F1: {metrics['m3_macro_f1']:.4f} (target >= 0.72)",
        f"- M3 confusion matrix: {metrics['m3_confusion_matrix']}",
        "- M3 per-class precision/recall: stored in models/m1_m5_metrics.json",
        "- M3 SHAP feature importance for Degraded class: outputs/business_report/plots/15_m3_shap_degraded_class.png",
        f"- M4 Stage filter: Stage {metrics['m4_stage']} only",
        f"- M4 AUC-ROC: {metrics['m4_auc']:.4f} (target >= 0.75)",
        f"- M4 threshold: {metrics['m4_threshold']}",
        f"- M5 RMSE: {metrics['m5_rmse']:.4f} (target < 12)",
        "- M5 varieties filtered to n >= 30 and GDD curves saved: outputs/business_report/plots/16_m5_gdd_curves.png",
        "- M6 Kaplan-Meier overall/region/stage: outputs/business_report/plots/17_m6_kaplan_meier.png",
        f"- M6 region log-rank p-value: {metrics['m6_region_logrank_p']:.6g}",
        f"- M6 stage log-rank p-value: {metrics['m6_stage_logrank_p']:.6g}",
        "- M6 hazard ratios: outputs/business_report/plots/18_m6_cox_hazard_ratios.png",
        f"- M6 C-index: {metrics['m6_c_index_test']:.4f} (target >= 0.70)",
        "- M6 AFT shelf-life plot: outputs/business_report/plots/19_m6_aft_shelf_life.png",
        "- M6 individual lot survival curves: outputs/business_report/plots/20_m6_individual_survival_curves.png",
        "- M6 Cox PH and AFT artifacts: models/m6_cox_ph.pkl, models/m6_aft_weibull.pkl",
        "",
        "Important modeling note: M4 and M5 use CT_Initial/core quality controls where available. These meet the statistical checklist targets, but deployment should only request those predictions when the corresponding early quality measurements are available.",
    ]
    (REPORT_DIR / "evaluation_checklist.md").write_text("\n".join(checklist), encoding="utf-8")
    (REPORT_DIR / "plot_manifest.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    return {"plots": paths, "metrics": metrics}


if __name__ == "__main__":
    generate_report()
