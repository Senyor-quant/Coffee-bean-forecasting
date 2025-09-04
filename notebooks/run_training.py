from pathlib import Path
import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

# -------------------------------
# STEP 1: LOAD, CLEAN, SCALE
# -------------------------------
csv_path = "/content/combined_data_arimax_05052025_1301.csv"  # keep your Colab path
df_lstm = pd.read_csv(csv_path)
df_lstm = df_lstm.drop(columns=['Time', 'Unnamed: 12', 'Unnamed: 13'], errors='ignore')
df_lstm.dropna(inplace=True)

# Final split: 80% train/val (CV), 20% holdout (test)
split_index = int(len(df_lstm) * 0.8)
df_cv = df_lstm.iloc[:split_index].copy()
df_holdout = df_lstm.iloc[split_index:].copy()

# Scale based on CV set only (avoid leakage)
scaler = MinMaxScaler()
scaler.fit(df_cv)

scaled_cv = pd.DataFrame(scaler.transform(df_cv), columns=df_cv.columns, index=df_cv.index)
scaled_holdout = pd.DataFrame(scaler.transform(df_holdout), columns=df_holdout.columns, index=df_holdout.index)

# -------------------------------
# STEP 2: CREATE SEQUENCES
# -------------------------------
n_steps = 30
target_column = 'KC_1'
target_idx = list(df_lstm.columns).index(target_column)  # define once, reuse later

def create_lstm_sequences(df, target_col, window):
    X_seq, y_seq = [], []
    vals = df.values
    tgt_idx = df.columns.get_loc(target_col)
    for i in range(window, len(df)):
        X_seq.append(vals[i - window:i, :])
        y_seq.append(vals[i, tgt_idx])
    return np.array(X_seq), np.array(y_seq)

X_cv, y_cv = create_lstm_sequences(scaled_cv, target_column, n_steps)
X_test, y_test = create_lstm_sequences(scaled_holdout, target_column, n_steps)

n_features = X_cv.shape[2]

# -------------------------------
# STEP 3: CROSS-VALIDATION
# -------------------------------
def make_model(n_steps, n_features):
    # Use explicit Input() to avoid the warning
    return Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])

tscv = TimeSeriesSplit(n_splits=5)
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), start=1):
    print(f"\n Fold {fold}")

    X_train, X_val = X_cv[train_idx], X_cv[val_idx]
    y_train, y_val = y_cv[train_idx], y_cv[val_idx]

    model = make_model(n_steps, n_features)
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[TqdmCallback(verbose=1), early_stop],
        verbose=0
    )

    # Predict & inverse-transform to original units
    y_val_pred = model.predict(X_val, verbose=0).flatten()

    zeros_val = np.zeros((len(y_val), n_features))
    zeros_val_pred = np.zeros_like(zeros_val)
    zeros_val[:, target_idx] = y_val
    zeros_val_pred[:, target_idx] = y_val_pred

    y_val_true = scaler.inverse_transform(zeros_val)[:, target_idx]
    y_val_pred_inv = scaler.inverse_transform(zeros_val_pred)[:, target_idx]

    rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred_inv)))
    print(f" Fold {fold} RMSE: {rmse:.2f}")
    cv_rmse_scores.append(rmse)

cv_mean = float(np.mean(cv_rmse_scores))
cv_std  = float(np.std(cv_rmse_scores))
print(f"\n CV Average RMSE: {cv_mean:.2f} ± {cv_std:.2f}")

# -------------------------------
# STEP 4: FINAL EVALUATION ON HOLDOUT
# -------------------------------
print("\n Evaluating on final holdout test set...")

# Retrain final model on full CV data
model_final = make_model(n_steps, n_features)
model_final.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_final.fit(
    X_cv, y_cv,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[TqdmCallback(verbose=1), early_stop],
    verbose=0
)

# Predict on holdout test in scaled space
y_test_pred = model_final.predict(X_test, verbose=0).flatten()

# Inverse-transform holdout target back to price units
zeros_test = np.zeros((len(y_test), n_features))
zeros_test_pred = np.zeros_like(zeros_test)
zeros_test[:, target_idx] = y_test
zeros_test_pred[:, target_idx] = y_test_pred

y_test_true = scaler.inverse_transform(zeros_test)[:, target_idx]
y_test_pred_inv = scaler.inverse_transform(zeros_test_pred)[:, target_idx]

rmse_test = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred_inv)))
print(f"\n Holdout Test RMSE: {rmse_test:.2f}")

# -------------------------------
# STEP 5: SAVE ARTIFACTS TO COLAB
# -------------------------------
out_models = Path("/content/models"); out_models.mkdir(parents=True, exist_ok=True)
out_data   = Path("/content/data/processed"); out_data.mkdir(parents=True, exist_ok=True)

# 1) model
model_path = out_models / "lstm_model.keras"
model_final.save(model_path)

# 2) metrics
metrics = {
    "cv_rmse_scores": cv_rmse_scores,
    "cv_rmse_mean": cv_mean,
    "cv_rmse_std": cv_std,
    "holdout_rmse": rmse_test,
    "n_steps": n_steps,
    "target_column": target_column,
    "input_file": csv_path
}
with open(out_data / "lstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 3) holdout predictions
pd.DataFrame({
    "y_test_true": y_test_true,
    "y_test_pred": y_test_pred_inv
}).to_csv(out_data / "lstm_holdout_predictions.csv", index=False)

print("\nSaved:")
print(" - model:", model_path)
print(" - metrics:", out_data / "lstm_metrics.json")
print(" - predictions:", out_data / "lstm_holdout_predictions.csv")
