# CDS-492 Team 3c: ML-Based Type 2 Diabetes Screening

Machine learning-based screening for undiagnosed Type 2 Diabetes using the 2024 South Korean National Health Insurance Service (NHIS) General Health Examination dataset.

**Course:** CDS-492 Capstone in Data Science, George Mason University, College of Science

**Authors:** Tyson Johnson (Team Lead), Alizeh Murtaza, Nithila Neethi Devan, Tariq Abdulhak

**Faculty Advisor:** Dr. Mohamed Adel Slamani

## Project Structure

- `code/t2d-screener/` -- Canonical codebase: CatBoost model, FastAPI backend, web frontend, SHAP explanations. See its [README](code/t2d-screener/README.md) and [MODEL_CARD](code/t2d-screener/MODEL_CARD.md) for details.
- `code/notebooks/` -- Supplementary analysis notebooks (9-experiment walkthrough, 8-model ROC comparison).
- `paper/` -- IEEE conference paper (LaTeX source, compiled PDF, figures).
- `presentations/` -- Final poster, comprehensive slide deck, and weekly module submissions.
- `docs/` -- Data dictionary and column descriptions.
- `_meta/` -- Agent reports and planning artifacts used during project consolidation.
- `_archive/` -- Full copies of original source folders and raw data files (not tracked in git).

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | CatBoost (40 features) |
| Test ROC-AUC | 0.8164 |
| Test PR-AUC | 0.2734 |
| Brier Score | 0.0643 |
| Screening Threshold | 0.0659 (tuned for 85%+ sensitivity) |

## Data Source

2024 NHIS General Health Examination dataset (~1M anonymized patient records):
https://www.data.go.kr/en/data/15007122/fileData.do

## Reproduction

See `code/t2d-screener/README.md` for setup and training instructions.

## License

MIT License. See [`code/t2d-screener/LICENSE`](code/t2d-screener/LICENSE).
