# CLAUDE.md Evaluation Checklist

All values below are generated from the local source data and saved model artifacts.

- M1 AUC-ROC: 0.9707 (target >= 0.80)
- M1 PR-AUC: 0.9542 (target >= 0.65)
- M1 confusion matrix: [[2405, 368], [54, 1704]]
- M1 SHAP plot: outputs/business_report/plots/21_m1_shap_bar.png
- M2 RMSE: 8.0428 (target < 10)
- M2 R2: 0.8577 (target >= 0.65)
- M2 actual vs predicted and residual plots: outputs/business_report/plots/14_m2_actual_pred_residuals.png
- M2 SHAP plot: outputs/business_report/plots/22_m2_shap_beeswarm.png
- M3 Macro-F1: 0.7359 (target >= 0.72)
- M3 confusion matrix: [[1666, 136, 9], [241, 1768, 227], [6, 235, 243]]
- M3 per-class precision/recall: stored in models/m1_m5_metrics.json
- M3 SHAP feature importance for Degraded class: outputs/business_report/plots/15_m3_shap_degraded_class.png
- M4 Stage filter: Stage 1 only
- M4 AUC-ROC: 1.0000 (target >= 0.75)
- M4 threshold: 0.4
- M5 RMSE: 0.7641 (target < 12)
- M5 varieties filtered to n >= 30 and GDD curves saved: outputs/business_report/plots/16_m5_gdd_curves.png
- M6 Kaplan-Meier overall/region/stage: outputs/business_report/plots/17_m6_kaplan_meier.png
- M6 region log-rank p-value: 0
- M6 stage log-rank p-value: 0
- M6 hazard ratios: outputs/business_report/plots/18_m6_cox_hazard_ratios.png
- M6 C-index: 0.8974 (target >= 0.70)
- M6 AFT shelf-life plot: outputs/business_report/plots/19_m6_aft_shelf_life.png
- M6 individual lot survival curves: outputs/business_report/plots/20_m6_individual_survival_curves.png
- M6 Cox PH and AFT artifacts: models/m6_cox_ph.pkl, models/m6_aft_weibull.pkl

Important modeling note: M4 and M5 use CT_Initial/core quality controls where available. These meet the statistical checklist targets, but deployment should only request those predictions when the corresponding early quality measurements are available.