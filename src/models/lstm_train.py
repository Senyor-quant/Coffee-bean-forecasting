import os
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------------------
# Sequence helper
# -------------------------------
def create_lstm_sequences(df: pd.DataFrame, target_col: str, window: int):
    X_seq, y_seq = [], []
    values = df.values
    target_idx = df.columns.get_loc(target_col)
    for i in range(window, len(df)):
        X_seq.append(values[i - window:i, :])
        y_seq.append(values[i, target_idx])
    return np.array(X_seq), np.array(y_seq)

# -------------------------------
# Model factory
# -------------------------------
def make_model(n_steps: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(n_steps, n_features)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------------
# Main training/evaluation
# -------------------------------
def run_training(
    data_dir: Path,
    input_file: str,
    target_column: str = "KC_1",
    n_steps: int = 30,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    verbose_fit: int = 0
):
    set_seed(seed)
    data_path = data_dir / input_file
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    # Load & clean
    df = pd.read_csv(data_path)
    df = df.drop(columns=['Time', 'Unnamed: 12', 'Unnamed: 13'], errors='ignore')
    df.dropna(inplace=True)

    # Final split: 80% CV, 20% holdout (time-ordered)
    split_index = int(len(df) * 0.8)
    df_cv = df.iloc[:split_index].copy()
    df_holdout = df.iloc[split_index:].copy()

    # Scale based on CV only (avoid leakage)
    scaler = MinMaxScaler()
    scaler.fit(df_cv)
    scaled_cv = pd.DataFrame(scaler.transform(df_cv), columns=df_cv.columns, index=df_cv.index)
    scaled_holdout = pd.DataFrame(scaler.transform(df_holdout), columns=df_holdout.columns, index=df_holdout.index)

    # Sequences
    X_cv, y_cv = create_lstm_sequences(scaled_cv, target_column, n_steps)
    X_test, y_test = create_lstm_sequences(scaled_holdout, target_column, n_steps)

    # CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmse_scores = []

    target_idx = df.columns.get_loc(target_column)
    n_features = X_cv.shape[2]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), start=1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_val = X_cv[train_idx], X_cv[val_idx]
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        model = make_model(n_steps, n_features)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[TqdmCallback(verbose=1), early_stop],
            verbose=verbose_fit
        )

        # Predict & inverse-transform
        y_val_pred = model.predict(X_val, verbose=0).flatten()

        zeros_val = np.zeros((len(y_val), n_features))
        zeros_val_pred = np.zeros_like(zeros_val)
        zeros_val[:, target_idx] = y_val
        zeros_val_pred[:, target_idx] = y_val_pred

        y_val_true = scaler.inverse_transform(zeros_val)[:, target_idx]
        y_val_pred_inv = scaler.inverse_transform(zeros_val_pred)[:, target_idx]

        rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred_inv)))
        print(f"Fold {fold} RMSE: {rmse:.4f}")
        cv_rmse_scores.append(rmse)

    cv_mean = float(np.mean(cv_rmse_scores))
    cv_std = float(np.std(cv_rmse_scores))
    print(f"\nCV Average RMSE: {cv_mean:.4f} ± {cv_std:.4f}")

    # Final fit on all CV data
    print("\nTraining final model on full CV data...")
    model_final = make_model(n_steps, n_features)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_final.fit(
        X_cv, y_cv,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[TqdmCallback(verbose=1), early_stop],
        verbose=verbose_fit
    )

    # Holdout evaluation
    print("\nEvaluating on holdout set...")
    y_test_pred = model_final.predict(X_test, verbose=0).flatten()

    zeros_test = np.zeros((len(y_test), n_features))
    zeros_test_pred = np.zeros_like(zeros_test)
    zeros_test[:, target_idx] = y_test
    zeros_test_pred[:, target_idx] = y_test_pred

    y_test_true = scaler.inverse_transform(zeros_test)[:, target_idx]
    y_test_pred_inv = scaler.inverse_transform(zeros_test_pred)[:, target_idx]
    rmse_test = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred_inv)))
    print(f"Holdout Test RMSE: {rmse_test:.4f}")

    # Save artifacts
    out_models = Path("models"); out_models.mkdir(parents=True, exist_ok=True)
    out_data = Path("data/processed"); out_data.mkdir(parents=True, exist_ok=True)

    model_path = out_models / "lstm_model.keras"
    model_final.save(model_path)

    metrics = {
        "cv_rmse_scores": cv_rmse_scores,
        "cv_rmse_mean": cv_mean,
        "cv_rmse_std": cv_std,
        "holdout_rmse": rmse_test,
        "n_steps": n_steps,
        "target_column": target_column,
        "input_file": str(data_path)
    }
    with open(out_data / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred_inv
    }).to_csv(out_data / "lstm_holdout_predictions.csv", index=False)

    return metrics, str(model_path)

def parse_args():
    ap = argparse.ArgumentParser(description="Train LSTM for coffee price forecasting")
    ap.add_argument("--data-dir", type=str, default=os.getenv("COFFEE_DATA_DIR", "./data"))
    ap.add_argument("--input-file", type=str, default=os.getenv("LSTM_INPUT_FILE", "combined_data_arimax_05052025_1301.csv"))
    ap.add_argument("--target", type=str, default=os.getenv("LSTM_TARGET_COLUMN", "KC_1"))
    ap.add_argument("--window", type=int, default=int(os.getenv("LSTM_WINDOW", 30)))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose-fit", type=int, default=0)
    return ap.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    metrics, model_path = run_training(
        data_dir=Path(args.data_dir),
        input_file=args.input_file,
        target_column=args.target,
        n_steps=args.window,
        n_splits=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose_fit=args.verbose_fit
    )
    print("\nSaved:")
    print(" - metrics:", "data/processed/lstm_metrics.json")
    print(" - model:", model_path)
    print(" - predictions:", "data/processed/lstm_holdout_predictions.csv")
