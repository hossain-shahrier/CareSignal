# CareSignal Model Card

## Model description

Binary classifier estimating **30-day inpatient readmission risk** at hospital discharge.

- **Algorithm:** LightGBM inside a scikit-learn pipeline (median imputation + one-hot encoding)
- **Input features:** age, sex, length of stay, prior admissions, condition count, procedure count, days since last discharge
- **Output:** probability of readmission within 30 days

## Intended use

- Portfolio demonstration of FHIR → ML → API workflow
- Educational exploration of readmission labeling on synthetic data

## Out-of-scope uses

- Real clinical decision-making
- Deployment without validation on institution-specific data
- Use on non-inpatient or non-FHIR data without retraining

## Training data

- **Source:** Synthea-generated FHIR R4 transaction bundles
- **Reference cohort:** 6 synthetic patients in `data/reference/` (CI fixtures)
- **Full cohort:** generate externally via `synthea_dataset_fhir` (2k–5k patients recommended)

## Label definition

For each finished inpatient encounter:

1. Discharge time = encounter `period.end`
2. Label = 1 if another inpatient encounter starts within 30 days after discharge
3. Features use only data available at or before discharge (no future leakage)

## Limitations

- Synthetic patients do not reflect real population prevalence or comorbidity patterns
- Small reference cohort is for engineering tests, not performance claims
- Model performance must be re-evaluated on each new cohort

## Evaluation metrics

See `artifacts/evaluation.json` after running:

```bash
python scripts/run_all.py --fhir-dir data/reference --config config/train.yaml
```

Reported metrics include ROC-AUC, AUC-PR, Brier score, F1, and accuracy at a validation-tuned threshold (isotonic calibration optional).

## Ethical considerations

Readmission models can disproportionately affect vulnerable populations if deployed without fairness review. This project documents the pipeline only and is not intended for production clinical use.
