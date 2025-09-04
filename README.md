# Coffee-bean-forecasting
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NHIIyjhrDIBiXDmQen7yo-rMAaMBLbGF?usp=sharing#scrollTo=B8pgKRvmMLxN)

Quick Start: Run the LSTM Model

This repo includes an LSTM model to forecast Arabica coffee prices from weather & macro data.
New recruits can try it immediately using Google Colab (no local setup needed).

1. Run the notebook

Click the Colab badge above.

Upload the dataset (combined_data_arimax_05052025_1301.csv) when prompted.

Press Runtime → Run all to train and evaluate the model.

2. What happens

The notebook:

Splits data → 80% for cross-validation, 20% holdout.

Runs 5 time-series folds (simulating different train/validation splits).

Reports average CV RMSE (in-sample) and final Holdout RMSE (future unseen).

Saves outputs under /content/data/processed and /content/models.

3. Results

Metrics: /content/data/processed/lstm_metrics.json

Predictions: /content/data/processed/lstm_holdout_predictions.csv

Model file: /content/models/lstm_model.keras

At the end of the notebook, a plot of Actual vs Predicted on the holdout set is shown.

**Note:** The model doesn’t run on GitHub itself. GitHub is for code + docs.
Training runs in Colab (or locally, if you have Python/TensorFlow installed).
