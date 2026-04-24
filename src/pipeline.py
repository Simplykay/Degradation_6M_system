"""Data loading and enrichment pipeline for the degradation project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import CSV_FILES, DATA_DIR
from .features import apply_quality_rules, engineer_features


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the raw data directory used by scripts and notebooks."""
    path = Path(data_dir) if data_dir is not None else DATA_DIR
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    return path


def csv_paths(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Map dataset keys to expected CSV paths."""
    root = resolve_data_dir(data_dir)
    return {key: root / filename for key, filename in CSV_FILES.items()}


def load_csvs(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load available project CSVs and report missing optional/required files."""
    paths = csv_paths(data_dir)
    loaded: dict[str, pd.DataFrame] = {}

    print(f"Loading CSVs from {resolve_data_dir(data_dir)}")
    for key, path in paths.items():
        if not path.exists():
            level = "optional" if key == "fuzzy" else "missing"
            print(f"  {key:<12} {level}: {path.name}")
            continue
        df = pd.read_csv(path, low_memory=False)
        loaded[key] = df
        print(f"  {key:<12} {df.shape[0]:>9,} rows  {df.shape[1]:>4} cols")

    required = set(CSV_FILES) - {"fuzzy"}
    missing = sorted(required - set(loaded))
    if missing:
        raise FileNotFoundError(f"Missing required datasets: {missing}")

    return loaded


def build_quality_base(lineage: pd.DataFrame) -> pd.DataFrame:
    """Create the lineage-anchored base table with targets."""
    return engineer_features(apply_quality_rules(lineage))


def attach_field_operations(base: pd.DataFrame, cotton_s3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join lineage to cottons3 via FUZZY_PRCS_ORD_NBR/process_order_number."""
    cs3 = apply_quality_rules(cotton_s3)
    fields = [
        "process_order_number",
        "pa_feature_id",
        "state",
        "variety",
        "irrigation_type",
        "pa_year",
        "maczone",
        "pre_defol_dd_60_cap90",
        "post_defol_dd_60_cap90",
        "pre_defol_total_precipitation",
        "post_defol_total_precipitation",
        "season_length",
        "bales_per_module",
        "total_modules",
        "planting_date",
        "harvest_date",
        "defoliation_date",
        "twc_grid_id",
    ]
    available = [col for col in fields if col in cs3.columns]
    cs3_fields = cs3[available].drop_duplicates(subset="process_order_number").copy()

    enriched = base.copy()
    enriched["FUZZY_PRCS_ORD_NBR_int"] = pd.to_numeric(
        enriched.get("FUZZY_PRCS_ORD_NBR"), errors="coerce"
    )
    cs3_fields["FUZZY_PRCS_ORD_NBR_int"] = pd.to_numeric(
        cs3_fields["process_order_number"], errors="coerce"
    )
    cs3_fields = cs3_fields.dropna(subset=["FUZZY_PRCS_ORD_NBR_int"])
    cs3_fields = cs3_fields.drop(columns=["process_order_number"])

    enriched = enriched.merge(cs3_fields, on="FUZZY_PRCS_ORD_NBR_int", how="left")
    field_count = int(enriched["irrigation_type"].notna().sum()) if "irrigation_type" in enriched else 0
    print(f"Rows with field data: {field_count:,} ({field_count / max(len(enriched), 1) * 100:.1f}%)")

    field_enriched = enriched[enriched["irrigation_type"].notna()].copy()
    return engineer_features(enriched), engineer_features(field_enriched)


def attach_crop_metadata(field_enriched: pd.DataFrame, cotton_cs: pd.DataFrame, cotton_cs2: pd.DataFrame) -> pd.DataFrame:
    """Attach cottonCS and cottonCS2 metadata to the field-enriched branch."""
    out = field_enriched.copy()
    if "pa_feature_id" not in out.columns:
        print("Skipping cottonCS joins: pa_feature_id not available")
        return out

    cs = cotton_cs.drop(columns=["seeding_rate"], errors="ignore").copy()
    cs_fields = [
        "pa_feature_id",
        "row_spacing",
        "twc_grid_id",
        "season_length",
        "planting_date",
        "harvest_date",
        "rm",
    ]
    cs = cs[[col for col in cs_fields if col in cs.columns]].drop_duplicates("pa_feature_id")

    cs2 = cotton_cs2.drop(columns=["NAWF_3"], errors="ignore").copy()
    cs2_fields = [
        "pa_feature_id",
        "nodes_above_white_flower_1",
        "nodes_above_white_flower_2",
        "defoliation_1",
    ]
    cs2 = cs2[[col for col in cs2_fields if col in cs2.columns]].drop_duplicates("pa_feature_id")

    out = out.merge(cs.add_suffix("_ccs").rename(columns={"pa_feature_id_ccs": "pa_feature_id"}), on="pa_feature_id", how="left")
    out = out.merge(cs2.add_suffix("_ccs2").rename(columns={"pa_feature_id_ccs2": "pa_feature_id"}), on="pa_feature_id", how="left")
    print(f"Rows after crop metadata join: {len(out):,}")
    return engineer_features(out)


def attach_weather(field_enriched: pd.DataFrame, weather_main: pd.DataFrame, weather_cs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weather before joining it to lot-level lineage data."""
    out = field_enriched.copy()

    wm = weather_main.copy()
    if "date" in wm.columns:
        wm["date"] = pd.to_datetime(wm["date"], errors="coerce")
    key_cols = ["variety", "state", "pa_year"]
    for col in ["variety", "state"]:
        if col in wm.columns:
            wm[col] = wm[col].astype(str).str.strip().str.upper()
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.upper()

    wm_agg = wm.groupby(key_cols).agg(
        wm_cumulated_dd60=("cumulated_dd60", "max"),
        wm_avg_soil_moisture=("avg_soil_moisture", "mean"),
        wm_total_dd60=("dd_60", "sum"),
        wm_irrigation_type=("irrigation_type", "first"),
        wm_maczone=("maczone", "first"),
        wm_state=("state", "first"),
        wm_planting_date=("planting_date", "first"),
        wm_defoliation_date=("defoliation_date", "first"),
        wm_harvest_date=("harvest_date", "first"),
    ).reset_index()

    wcs_agg = weather_cs.groupby("twc_grid_id").agg(
        wcs_mean_vpd=("VPD", "mean"),
        wcs_max_vpd=("VPD", "max"),
        wcs_heat_stress_days=("HS_level", lambda x: (x > 0).sum()),
        wcs_severe_heat_days=("HS_level", lambda x: (x == 4).sum()),
        wcs_mean_soil_moisture=("avg_soil_moisture", "mean"),
        wcs_water_balance_sum=("water_balance", "sum"),
        wcs_deficit_days=("water_balance", lambda x: (x <= 0).sum()),
        wcs_mean_rh=("avg_relative_humidity", "mean"),
        wcs_total_dd60=("dd60", "sum"),
    ).reset_index()

    if set(key_cols).issubset(out.columns):
        out = out.merge(wm_agg, on=key_cols, how="left")
        print(f"wm_cumulated_dd60 coverage: {out['wm_cumulated_dd60'].notna().mean() * 100:.1f}%")

    if "twc_grid_id" in out.columns:
        out = out.merge(wcs_agg, on="twc_grid_id", how="left")
        print(f"wcs_mean_vpd coverage: {out['wcs_mean_vpd'].notna().mean() * 100:.1f}%")

    return engineer_features(out)


def build_model_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Build notebook-ready base, enriched, and field-enriched tables."""
    data = load_csvs(data_dir)
    base = build_quality_base(data["lineage"])
    enriched, field_enriched = attach_field_operations(base, data["cotton_s3"])
    field_enriched = attach_crop_metadata(field_enriched, data["cotton_cs"], data["cotton_cs2"])
    field_enriched = attach_weather(field_enriched, data["weather_main"], data["weather_cs"])
    return {
        "base": base,
        "enriched": engineer_features(enriched),
        "field_enriched": field_enriched,
        **data,
    }
