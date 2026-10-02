# Freight Spot Rate Prediction

An end-to-end machine learning pipeline built with LightGBM to forecast spot freight rates (`posted_rate`) across multi-equipment US lanes. The architecture reformulates the regression target into a domain-specific Rate-Per-Mile (RPM) setup, stabilizing loss variance across trip lengths.

## 1. Quickstart & Setup

```bash
# Clone repository
git clone <YOUR_REPO_URL>
cd "Freight Rate Prediction"

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 2. Pipeline Execution

Run all cells in `notebooks/freight_rate_prediction.ipynb`.

The notebook executes the workflow sequentially:

1. Feature Engineering & Imputation: Engineers 29 features—including origin-destination corridor keys (`lane`), trigonometric date cycles (`dow_sin`, `dow_cos`), payload density (`weight / distance`), and historical city centroid lookups for missing coordinates.
2. Temporal Validation: Trains on Jan–Aug 2025 and evaluates on Sep–Oct 2025 to test generalization on future unseen market conditions without look-ahead leakage.
3. Retraining & Predictions: Retrains the winning RPM model on the complete 48,000-row dataset, reconstructs dollar totals via `predicted_rpm * distance`, and exports `validation_predictions.csv` alongside `data/december-chart-inputs.csv`.
4. Automated Verification: Executes `score.py` directly at the end of the notebook via a subprocess call to validate outputs and generate the evaluation chart in `scorer_results/candidate_december.png`.

Optional: Run the scorer manually from the root directory at any time:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

## 3. Results Summary

### Out-of-Time Holdout Evaluation (Sep 1 – Oct 31, 2025)

Trained on 38,477 loads; evaluated on 9,523 out-of-time loads.

| Model Variant | Objective / Target | Holdout MAE | Holdout RMSE | R² Score |
| --- | --- | ---: | ---: | ---: |
| Baseline LightGBM | Direct Dollar Rate (`posted_rate`) | $125.16 | $639.02 | 0.8247 |
| Optimized LightGBM | Rate-Per-Mile ($\text{RPM} \times \text{distance}$) | $112.49 | $635.22 | 0.8267 |
| Net Gain | — | -$12.67 (-10.13%) | -$3.80 | +0.0020 |

### December Corridor Sensitivity Test (Lexington, KY → Fort Wayne, IN)

Evaluated on the fixed 360-mile Dry Van test lane across all 31 days of December 2025:

- Predicted Rate Range: $819 – $836 (~$2.28 – $2.32/mile), aligning with historical Midwest spot benchmarks.
- Observed Behavior: Accurately reflects 7-day cyclical oscillations (mid-week dispatch premiums vs. weekend volume dips) and captures the holiday capacity tightening surge heading into late December.
- Output Artifact: Visualized in `scorer_results/candidate_december.png`.

## 4. Repository Structure

```plaintext
.
├── data/
│   ├── december-chart-inputs.csv             # Updated December predictions (31 rows)
│   ├── train-test.csv                        # Historical training data (48k rows)
│   ├── validation.csv                        # Prospective validation loads (12k rows)
│   ├── validation-predictions-template.csv   # Target format template
│   └── validation_predictions.csv            # Backup predictions file
├── notebooks/
│   └── freight_rate_prediction.ipynb         # End-to-end pipeline notebook
├── scorer_results/
│   └── candidate_december.png                # Generated evaluation plot
├── .gitignore
├── README.md
├── requirements.txt
├── score.py                                  # Evaluation script
└── validation_predictions.csv                # Primary submission predictions (root level)
```

## 5. Submission Requirements

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach, and `candidate_december.png`
- 2–3 minute Loom link
