"""Project constants for cotton seed degradation modeling."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CT_THRESHOLD = 60
RANDOM_STATE = 42
CURRENT_YEAR = 2026

TRAIN_SEASONS = [2017, 2019, 2020, 2021]
VAL_SEASONS = [2022]
TEST_SEASONS = [2023, 2024]
HOLDOUT_SEASONS = [2025]

CORE_FEATURES = [
    "WG_Current",
    "CT_Initial",
    "Moisture",
    "Mechanical_Damage",
    "Actual_Seed_Per_LB",
    "Stage",
    "season_age",
    "RR_Lateral_Strip_PCT",
    "Cry1Ac_Bollgard_Strip_Test",
    "Cry2Ab_Bollgard_Strip_Test",
]

CAT_FEATURES = ["Variety", "Origin_Region", "Grower_Region"]

FIELD_FEATURES = [
    "irrigation_type",
    "irrigation_is_dryland",
    "season_length",
    "pre_defol_dd_60_cap90",
    "post_defol_dd_60_cap90",
    "pre_defol_total_precipitation",
    "defol_to_harvest",
    "defol_to_harvest_cum_dd60",
    "bales_per_module",
]

WEATHER_FEATURES = [
    "cumulated_dd60",
    "cumulated_soil_moisture",
    "heat_stress_days",
    "VPD_stress_score",
    "water_balance_deficit",
    "pp_day5_cum_dd60",
    "pp_day10_cum_dd60",
    "pp_day5_avg_soilmoisture",
    "pp_day10_avg_soilmoisture",
    "pp_day5_cum_soilmoisture",
    "pp_day10_cum_soilmoisture",
]

SURVIVAL_FEATURES = [
    "WG_Current",
    "CT_Initial",
    "Moisture",
    "Mechanical_Damage",
    "Actual_Seed_Per_LB",
    "Stage",
    "season_age",
    "ct_distance_to_threshold",
    "pre_defol_dd_60_cap90",
    "irrigation_is_dryland",
    "cumulated_dd60",
    "cumulated_soil_moisture",
]

CSV_FILES = {
    "lineage": "vw_cotton_lineage_and_quality_june_fg_all_cols.csv",
    "quality": "vw_cotton_qlty_rslt_all_cols.csv",
    "cotton_s3": "cottons3_2025.csv",
    "cotton_cs": "cottonCSs3_2025.csv",
    "cotton_cs2": "cottonCS2s3_2025.csv",
    "weather_main": "2026_cotton_with_weather.csv",
    "weather_cs": "weatherCSs3_2025.csv",
    "weather_s3": "weathers3_2025.csv",
    "fuzzy": "fuzzy_lineage_query.csv",
}
