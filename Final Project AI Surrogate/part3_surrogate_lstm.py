import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber


# =========================================================
# PART 3 - SURROGATE MODEL
# Recommended version:
# - many-to-one only
# - 3 separate models
# - recursive rollout for full trajectory
# - uses past values of all 3 variables + static metadata
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "gama_runs"
META_FILE = DATA_DIR / "run_metadata.csv"
OUT_DIR = BASE_DIR / "part3_outputs_many_to_one_3models"

TARGETS = ["num_changes", "variance_scores", "social_welfare"]
TARGET_LABELS = {
    "num_changes": "Strategic opinion changes",
    "variance_scores": "Variance scores",
    "social_welfare": "Social welfare"
}

LOG_TARGETS = {"num_changes", "variance_scores"}
LOSS_BY_TARGET = {
    "num_changes": Huber(delta=1.0),
    "variance_scores": Huber(delta=1.0),
    "social_welfare": "mse"
}

HORIZON = 61           # days 0..60
WINDOW = 7             # many-to-one input window
SEQ_DIM = 4            # num_changes, variance_scores, social_welfare, day_norm

TEST_SIZE = 0.2
VAL_SIZE = 0.2
SEED = 42

EPOCHS = 120
BATCH_SIZE = 16
LR = 3e-4
N_TEST_PLOTS = 3
EXTRAP_N = 100000

np.random.seed(SEED)
tf.random.set_seed(SEED)

for p in [OUT_DIR, OUT_DIR / "csv", OUT_DIR / "plots", OUT_DIR / "models"]:
    p.mkdir(parents=True, exist_ok=True)


def make_run_id(s, n, r):
    return f"{s}_N{int(n)}_run_{int(r):03d}"


def load_runs():
    meta = pd.read_csv(META_FILE).set_index("run_folder").to_dict("index")
    runs, merged = [], []

    for f in sorted(DATA_DIR.glob("simulation_results_*.csv")):
        df = pd.read_csv(f).sort_values("day").reset_index(drop=True)

        rid = make_run_id(
            df.loc[0, "scenario_id"],
            df.loc[0, "num_agents"],
            df.loc[0, "repetition_id"]
        )

        if rid not in meta:
            raise ValueError(f"No metadata match for {f.name} -> {rid}")
        if len(df) != HORIZON:
            raise ValueError(f"{f.name}: expected {HORIZON} rows, got {len(df)}")

        m = meta[rid]

        static = {
            "num_agents_log": np.log1p(m["num_agents"]),
            "prop_stubborn": float(m["prop_stubborn"]),
            "prop_strategic": float(m["prop_strategic"]),
            "prop_mixed": float(m["prop_mixed"]),
            "avg_degree": float(m["avg_degree"]),
            "network_type": str(m["network_type"]),
            "preference_model": str(m["preference_model"]),
        }

        y = df[TARGETS].astype(float).to_numpy()

        runs.append({
            "run_id": rid,
            "days": df["day"].astype(int).to_numpy(),
            "y": y,
            "static": static
        })

        merged.append(df.assign(run_id=rid, avg_degree=m["avg_degree"], seed=m["seed"]))

    if not runs:
        raise FileNotFoundError(f"No simulation_results_*.csv files found in {DATA_DIR}")

    pd.concat(merged, ignore_index=True).to_csv(
        OUT_DIR / "csv" / "merged_long_dataset.csv", index=False
    )
    return runs


def fit_static_preprocessors(train_runs):
    s = pd.DataFrame([r["static"] for r in train_runs])

    num_cols = ["num_agents_log", "prop_stubborn", "prop_strategic", "prop_mixed", "avg_degree"]
    cat_cols = ["network_type", "preference_model"]

    x_scaler = StandardScaler().fit(s[num_cols])

    try:
        x_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        x_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    x_encoder.fit(s[cat_cols])
    return x_scaler, x_encoder


def static_vector(static_dict, x_scaler, x_encoder):
    df = pd.DataFrame([static_dict])

    num = x_scaler.transform(df[["num_agents_log", "prop_stubborn", "prop_strategic", "prop_mixed", "avg_degree"]])
    cat = x_encoder.transform(df[["network_type", "preference_model"]])

    return np.concatenate([num, cat], axis=1).astype(np.float32).flatten()


def forward_transform(target_name, values):
    values = np.asarray(values, dtype=float)
    if target_name in LOG_TARGETS:
        values = np.log1p(np.clip(values, 0.0, None))
    return values


def inverse_transform(target_name, values):
    values = np.asarray(values, dtype=float)
    if target_name in LOG_TARGETS:
        values = np.expm1(values)
        values = np.clip(values, 0.0, None)
    return values


def fit_target_scalers(train_runs):
    scalers = {}
    for j, name in enumerate(TARGETS):
        vals = np.concatenate([forward_transform(name, r["y"][:, j]) for r in train_runs])
        scalers[name] = StandardScaler().fit(vals.reshape(-1, 1))
    return scalers


def scale_target(target_name, values, target_scalers):
    v = forward_transform(target_name, values).reshape(-1, 1)
    return target_scalers[target_name].transform(v).reshape(-1)


def inverse_scale_target(target_name, values_scaled, target_scalers):
    v = np.asarray(values_scaled, dtype=float).reshape(-1, 1)
    v = target_scalers[target_name].inverse_transform(v).reshape(-1)
    return inverse_transform(target_name, v)


def pad_history(y_hist, d_hist, window):
    if len(y_hist) == 0:
        raise ValueError("History cannot be empty.")

    if len(y_hist) >= window:
        return y_hist[-window:], d_hist[-window:]

    pad_n = window - len(y_hist)
    y_pad = np.repeat(y_hist[[0]], pad_n, axis=0)
    d_pad = np.repeat(d_hist[[0]], pad_n)

    return np.vstack([y_pad, y_hist]), np.concatenate([d_pad, d_hist])


def make_seq_features(y_hist, d_hist, target_scalers):
    y_pad, d_pad = pad_history(y_hist, d_hist, WINDOW)

    num_changes_scaled = scale_target("num_changes", y_pad[:, 0], target_scalers)
    variance_scaled = scale_target("variance_scores", y_pad[:, 1], target_scalers)
    social_scaled = scale_target("social_welfare", y_pad[:, 2], target_scalers)
    day_norm = d_pad.astype(float) / (HORIZON - 1)

    seq = np.column_stack([
        num_changes_scaled,
        variance_scaled,
        social_scaled,
        day_norm
    ]).astype(np.float32)

    return seq


def build_supervised_data(runs, x_scaler, x_encoder, target_scalers, target_name):
    j = TARGETS.index(target_name)

    X_seq, X_static, y, meta_rows = [], [], [], []

    for r in runs:
        svec = static_vector(r["static"], x_scaler, x_encoder)
        series = r["y"]
        days = r["days"]

        for t in range(1, HORIZON):
            seq = make_seq_features(series[:t], days[:t], target_scalers)
            yt = scale_target(target_name, np.array([series[t, j]]), target_scalers)[0]

            X_seq.append(seq)
            X_static.append(svec)
            y.append(yt)
            meta_rows.append({"run_id": r["run_id"], "day": int(days[t])})

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_static, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1, 1),
        pd.DataFrame(meta_rows)
    )


def build_model(seq_dim, static_dim, loss_fn):
    seq_in = Input(shape=(WINDOW, seq_dim), name="seq_input")
    x = LSTM(96, return_sequences=True)(seq_in)
    x = Dropout(0.20)(x)
    x = LSTM(32)(x)

    static_in = Input(shape=(static_dim,), name="static_input")
    s = Dense(32, activation="relu")(static_in)

    z = Concatenate()([x, s])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.20)(z)
    z = Dense(32, activation="relu")(z)
    out = Dense(1, name="target")(z)

    model = Model(inputs=[seq_in, static_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss=loss_fn,
        metrics=["mae"]
    )
    return model


def metrics_1d(actual, pred):
    return {
        "mse": float(mean_squared_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "r2": float(r2_score(actual, pred))
    }


def plot_training(history, path, target_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training history: {target_name}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_curve(days, actual, pred, path, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(days, actual, label="Real data (GAMA)", lw=2.5)
    plt.plot(days, pred, label="Predicted (surrogate)", lw=2.5)
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Prediction starts after day 0")
    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_heatmap(run_ids, days, actual, pred, path, title):
    err = np.abs(pred - actual)
    plt.figure(figsize=(12, max(4, 0.4 * len(run_ids))))
    plt.imshow(err, aspect="auto", cmap="magma")
    plt.colorbar(label="Absolute error")
    plt.yticks(range(len(run_ids)), run_ids)
    plt.xticks(range(len(days)), days)
    plt.xlabel("Day")
    plt.ylabel("Test run")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def rollout_run(run, models, x_scaler, x_encoder, target_scalers, static_override=None):
    days = run["days"]
    actual = run["y"].copy()
    pred = np.zeros_like(actual, dtype=float)

    pred[0] = actual[0]  # seed day 0 with true initial state

    static_source = static_override if static_override is not None else run["static"]
    svec = static_vector(static_source, x_scaler, x_encoder)[None, :]

    for t in range(1, HORIZON):
        seq = make_seq_features(pred[:t], days[:t], target_scalers)[None, :, :]

        next_vals = []
        for name in TARGETS:
            yhat_scaled = models[name].predict([seq, svec], verbose=0).reshape(-1)
            yhat = inverse_scale_target(name, yhat_scaled, target_scalers)[0]
            if name in LOG_TARGETS:
                yhat = max(0.0, yhat)
            next_vals.append(float(yhat))

        pred[t] = next_vals

    return pred


def save_test_curves(test_runs, preds, path):
    rows = []
    for r, pred in zip(test_runs, preds):
        actual = r["y"]
        for t, d in enumerate(r["days"]):
            rows.append({
                "run_id": r["run_id"],
                "day": int(d),
                "actual_num_changes": float(actual[t, 0]),
                "predicted_num_changes": float(pred[t, 0]),
                "actual_variance_scores": float(actual[t, 1]),
                "predicted_variance_scores": float(pred[t, 1]),
                "actual_social_welfare": float(actual[t, 2]),
                "predicted_social_welfare": float(pred[t, 2]),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    runs = load_runs()

    ids = [r["run_id"] for r in runs]
    train_val_ids, test_ids = train_test_split(ids, test_size=TEST_SIZE, random_state=SEED)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio, random_state=SEED)

    train_runs = [r for r in runs if r["run_id"] in train_ids]
    val_runs = [r for r in runs if r["run_id"] in val_ids]
    test_runs = [r for r in runs if r["run_id"] in test_ids]

    x_scaler, x_encoder = fit_static_preprocessors(train_runs)
    target_scalers = fit_target_scalers(train_runs)

    static_dim = len(static_vector(train_runs[0]["static"], x_scaler, x_encoder))
    models = {}
    histories = {}

    for target_name in TARGETS:
        X_seq_train, X_static_train, y_train, _ = build_supervised_data(
            train_runs, x_scaler, x_encoder, target_scalers, target_name
        )
        X_seq_val, X_static_val, y_val, _ = build_supervised_data(
            val_runs, x_scaler, x_encoder, target_scalers, target_name
        )

        model = build_model(SEQ_DIM, static_dim, LOSS_BY_TARGET[target_name])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
            ModelCheckpoint(
                str(OUT_DIR / "models" / f"best_{target_name}.keras"),
                monitor="val_loss",
                save_best_only=True
            )
        ]

        history = model.fit(
            [X_seq_train, X_static_train],
            y_train,
            validation_data=([X_seq_val, X_static_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        models[target_name] = model
        histories[target_name] = history

        plot_training(
            history,
            OUT_DIR / "plots" / f"training_history_{target_name}.png",
            target_name
        )

    # Recursive test rollout
    test_preds = []
    for r in test_runs:
        pred = rollout_run(r, models, x_scaler, x_encoder, target_scalers)
        test_preds.append(pred)

    save_test_curves(test_runs, test_preds, OUT_DIR / "csv" / "test_curves.csv")

    # Metrics from day 1 onward because day 0 is the seed
    metrics_by_target = {}
    for j, target_name in enumerate(TARGETS):
        actual = np.concatenate([r["y"][1:, j] for r in test_runs])
        pred = np.concatenate([p[1:, j] for p in test_preds])
        metrics_by_target[target_name] = metrics_1d(actual, pred)

    summary = {
        "model_family": "three separate many-to-one LSTM surrogates",
        "targets": TARGETS,
        "horizon_days": HORIZON,
        "window": WINDOW,
        "seed_info": "day 0 is used as the initial seed; days 1..60 are predicted recursively",
        "inputs_used": [
            "past num_changes",
            "past variance_scores",
            "past social_welfare",
            "day index",
            "num_agents",
            "network_type",
            "preference_model",
            "prop_stubborn",
            "prop_strategic",
            "prop_mixed",
            "avg_degree"
        ],
        "training_params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "clipnorm": 1.0
        },
        "split_sizes": {
            "train_runs": len(train_runs),
            "val_runs": len(val_runs),
            "test_runs": len(test_runs)
        },
        "test_metrics_day1_to_60": metrics_by_target
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Example curves
    for i, r in enumerate(test_runs[:N_TEST_PLOTS], start=1):
        pred = test_preds[i - 1]
        for j, target_name in enumerate(TARGETS):
            plot_curve(
                r["days"],
                r["y"][:, j],
                pred[:, j],
                OUT_DIR / "plots" / f"real_vs_predicted_{target_name}_{i}_{r['run_id']}.png",
                f"Test run {r['run_id']}: real vs predicted ({target_name})",
                TARGET_LABELS[target_name]
            )

    # Heatmaps day 1..60
    run_ids = [r["run_id"] for r in test_runs]
    days = test_runs[0]["days"][1:]

    for j, target_name in enumerate(TARGETS):
        actual_mat = np.array([r["y"][1:, j] for r in test_runs])
        pred_mat = np.array([p[1:, j] for p in test_preds])

        plot_heatmap(
            run_ids,
            days,
            actual_mat,
            pred_mat,
            OUT_DIR / "plots" / f"error_heatmap_{target_name}.png",
            f"Prediction error heatmap: {target_name}"
        )

    # Extrapolation to 100,000 agents
    base_run = test_runs[0]
    extra_static = dict(base_run["static"])
    extra_static["num_agents_log"] = np.log1p(EXTRAP_N)

    extra_pred = rollout_run(
        base_run,
        models,
        x_scaler,
        x_encoder,
        target_scalers,
        static_override=extra_static
    )

    pd.DataFrame({
        "day": base_run["days"],
        "predicted_num_changes_100000_agents": extra_pred[:, 0],
        "predicted_variance_scores_100000_agents": extra_pred[:, 1],
        "predicted_social_welfare_100000_agents": extra_pred[:, 2],
    }).to_csv(OUT_DIR / "csv" / "extrapolation_100000_agents.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for j, target_name in enumerate(TARGETS):
        axes[j].plot(base_run["days"], extra_pred[:, j], lw=2.5)
        axes[j].set_ylabel(TARGET_LABELS[target_name])
        axes[j].set_title(f"Extrapolation to 100,000 agents: {target_name}")
        axes[j].grid(alpha=0.25)
    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plots" / "extrapolation_100000_agents.png", dpi=220)
    plt.close()

    print("Done.")
    print("Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
