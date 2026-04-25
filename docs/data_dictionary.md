# Data Dictionary: NHIS 2024 General Health Examination

## Source

- **Dataset:** National Health Insurance Service (NHIS) General Health Examination, 2024
- **Portal:** Korea Open Data Portal
- **URL:** https://www.data.go.kr/en/data/15007122/fileData.do
- **Records:** ~1,000,000 anonymized patient records
- **Columns:** 33 raw fields
- **Encoding:** cp949 (Korean)

## Access Instructions

1. Visit the URL above
2. Download the CSV file (approximately 92 MB)
3. Place it at `code/t2d-screener/data/health_2024.csv`
4. The training script (`train.py`) can also download it automatically from the GitHub mirror

## Column Descriptions

See `nhis_dataset_column_descriptions.xlsx` in this directory for the full Korean-to-English column mapping with descriptions, units, and value ranges.

For the canonical column schema used in the ML pipeline, see `code/t2d-screener/src/config.py` which defines all column names, validation domains, and feature engineering specifications.

## Feature Groups (40 total)

1. **Core Numeric (10):** age, height, weight, waist, systolic_bp, diastolic_bp, bmi, waist_to_height, ww_index, pulse_pressure
2. **Core Nominal (5):** sex, smoking_status, drinking_freq, exercise_freq, hearing_left (+ blindness flags)
3. **Non-Lipid Labs (11):** hemoglobin, fpg, sgot_ast, sgpt_alt, ggt, serum_creatinine, ast_alt_ratio, log_ggt, log_creatinine, mean_arterial_pressure, has_non_lipid_labs
4. **Lipid Panel (14):** total_cholesterol, hdl, ldl, triglycerides, tc_hdl_ratio, tg_hdl_ratio, ldl_hdl_ratio, non_hdl_cholesterol, log_triglycerides, has_lipid_panel, + missingness flags
