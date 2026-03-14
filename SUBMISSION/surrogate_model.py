import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber

# Global config
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

HORIZON = 61
WINDOW = 7
SEQ_DIM = 4

TEST_SIZE = 0.2
VAL_SIZE = 0.2
SEED = 42
EPOCHS = 120
BATCH_SIZE = 16
LR = 3e-4
EXTRAP_N = 100000

np.random.seed(SEED)
tf.random.set_seed(SEED)

for p in [OUT_DIR, OUT_DIR / "csv", OUT_DIR / "plots", OUT_DIR / "models"]:
    p.mkdir(parents=True, exist_ok=True)

def load_dataset():
    meta_df = pd.read_csv(META_FILE)
    meta_map = meta_df.set_index("run_folder").to_dict("index")
    runs, merged = [], []
    files = sorted(DATA_DIR.glob("simulation_results_*.csv"))
    for f in files:
        try:
            df = pd.read_csv(f).sort_values("day").reset_index(drop=True)
            rid = f.name.replace("simulation_results_", "").replace(".csv", "")
            if rid not in meta_map or len(df) != HORIZON: continue
            m = meta_map[rid]
            static_feats = {
                "num_agents_log": np.log1p(m["num_agents"]),
                "prop_stubborn": float(m["prop_stubborn"]),
                "prop_strategic": float(m["prop_strategic"]),
                "prop_mixed": float(m["prop_mixed"]),
                "avg_degree": float(m["avg_degree"]),
                "network_type": str(m["network_type"]),
                "preference_model": str(m["preference_model"]),
            }
            y_vals = df[TARGETS].astype(float).to_numpy()
            runs.append({"run_id": rid, "days": df["day"].astype(int).to_numpy(), "y": y_vals, "static": static_feats})
            merged.append(df.assign(run_id=rid))
        except: continue
    if merged:
        pd.concat(merged, ignore_index=True).to_csv(OUT_DIR / "csv" / "merged_long_dataset.csv", index=False)
    return runs

def setup_preprocessors(runs):
    s_df = pd.DataFrame([r["static"] for r in runs])
    num_cols = ["num_agents_log", "prop_stubborn", "prop_strategic", "prop_mixed", "avg_degree"]
    cat_cols = ["network_type", "preference_model"]
    scaler = StandardScaler().fit(s_df[num_cols])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(s_df[cat_cols])
    return scaler, encoder

def get_static_vec(static_dict, scaler, encoder):
    df = pd.DataFrame([static_dict])
    num = scaler.transform(df[["num_agents_log", "prop_stubborn", "prop_strategic", "prop_mixed", "avg_degree"]])
    cat = encoder.transform(df[["network_type", "preference_model"]])
    return np.concatenate([num, cat], axis=1).flatten().astype(np.float32)

def apply_transforms(target, vals):
    vals = np.asarray(vals, dtype=float)
    if target in LOG_TARGETS: return np.log1p(np.clip(vals, 0.0, None))
    return vals

def reverse_transforms(target, vals):
    vals = np.asarray(vals, dtype=float)
    if target in LOG_TARGETS: return np.clip(np.expm1(vals), 0.0, None)
    return vals

def init_target_scalers(runs):
    scalers = {}
    for i, name in enumerate(TARGETS):
        v = np.concatenate([apply_transforms(name, r["y"][:, i]) for r in runs])
        scalers[name] = StandardScaler().fit(v.reshape(-1, 1))
    return scalers

def inverse_scale(name, vals, scalers):
    v = np.asarray(vals, dtype=float).reshape(-1, 1)
    v = scalers[name].inverse_transform(v).reshape(-1)
    return reverse_transforms(name, v)

def pad_seq(y, d):
    if len(y) >= WINDOW: return y[-WINDOW:], d[-WINDOW:]
    pad_len = WINDOW - len(y)
    y_pad = np.repeat(y[[0]], pad_len, axis=0)
    d_pad = np.repeat(d[[0]], pad_len)
    return np.vstack([y_pad, y]), np.concatenate([d_pad, d])

def get_seq_features(y_hist, d_hist, scalers):
    y_p, d_p = pad_seq(y_hist, d_hist)
    c_s = scalers["num_changes"].transform(apply_transforms("num_changes", y_p[:, 0]).reshape(-1, 1)).flatten()
    v_s = scalers["variance_scores"].transform(apply_transforms("variance_scores", y_p[:, 1]).reshape(-1, 1)).flatten()
    s_s = scalers["social_welfare"].transform(apply_transforms("social_welfare", y_p[:, 2]).reshape(-1, 1)).flatten()
    d_n = d_p.astype(float) / (HORIZON - 1)
    return np.column_stack([c_s, v_s, s_s, d_n]).astype(np.float32)

def build_train_data(runs, scaler, encoder, scalers, target):
    idx = TARGETS.index(target)
    X_seq, X_static, y_lab = [], [], []
    for r in runs:
        svec = get_static_vec(r["static"], scaler, encoder)
        for t in range(1, HORIZON):
            X_seq.append(get_seq_features(r["y"][:t], r["days"][:t], scalers))
            X_static.append(svec)
            y_t = scalers[target].transform(apply_transforms(target, [r["y"][t, idx]]).reshape(-1, 1))[0]
            y_lab.append(y_t)
    return np.array(X_seq), np.array(X_static), np.array(y_lab).reshape(-1, 1)

def compile_lstm(static_dim, loss_fn):
    seq_in = Input(shape=(WINDOW, SEQ_DIM))
    x = LSTM(96, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    st_in = Input(shape=(static_dim,))
    s = Dense(32, activation="relu")(st_in)
    z = Concatenate()([x, s])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.2)(z)
    z = Dense(32, activation="relu")(z)
    out = Dense(1)(z)
    model = Model(inputs=[seq_in, st_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR, clipnorm=1.0), loss=loss_fn, metrics=["mae"])
    return model

def predict_rollout(run, models, scaler, encoder, scalers, override=None):
    pred = np.zeros_like(run["y"])
    pred[0] = run["y"][0]
    s_info = override if override else run["static"]
    svec = get_static_vec(s_info, scaler, encoder)[None, :]
    for t in range(1, HORIZON):
        seq = get_seq_features(pred[:t], run["days"][:t], scalers)[None, :]
        step_res = []
        for name in TARGETS:
            yhat = models[name].predict([seq, svec], verbose=0).flatten()
            val = inverse_scale(name, yhat, scalers)[0]
            if name in LOG_TARGETS: val = max(0.0, val)
            step_res.append(val)
        pred[t] = step_res
    return pred

def main():
    runs = load_dataset()
    ids = [r["run_id"] for r in runs]
    tr_val_ids, te_ids = train_test_split(ids, test_size=TEST_SIZE, random_state=SEED)
    tr_ids, val_ids = train_test_split(tr_val_ids, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=SEED)
    tr_runs = [r for r in runs if r["run_id"] in tr_ids]
    vl_runs = [r for r in runs if r["run_id"] in val_ids]
    te_runs = [r for r in runs if r["run_id"] in te_ids]

    scaler, encoder = setup_preprocessors(tr_runs)
    t_scalers = init_target_scalers(tr_runs)
    s_dim = len(get_static_vec(tr_runs[0]["static"], scaler, encoder))
    
    models = {}
    for name in TARGETS:
        xs, xst, ys = build_train_data(tr_runs, scaler, encoder, t_scalers, name)
        vxs, vxst, vys = build_train_data(vl_runs, scaler, encoder, t_scalers, name)
        m = compile_lstm(s_dim, LOSS_BY_TARGET[name])
        m.fit([xs, xst], ys, validation_data=([vxs, vxst], vys), epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[EarlyStopping(patience=15, restore_best_weights=True)], verbose=1)
        models[name] = m

    # Extrapolation Comparison Plot
    base = te_runs[0]
    extra_static = dict(base["static"])
    extra_static["num_agents_log"] = np.log1p(EXTRAP_N)
    pred_100k = predict_rollout(base, models, scaler, encoder, t_scalers, override=extra_static)
    
    # Normalize for single graph comparison
    mms = MinMaxScaler()
    norm_pred = mms.fit_transform(pred_100k)
    
    plt.figure(figsize=(10, 6))
    plt.plot(base["days"], norm_pred[:, 0], label="Opinion Changes", color="red", lw=2.5)
    plt.plot(base["days"], norm_pred[:, 1], label="Variance (Consensus)", color="blue", lw=2.5)
    plt.plot(base["days"], norm_pred[:, 2], label="Social Welfare", color="green", lw=2.5)
    plt.title("Relative Metric Evolution (Extrapolated to 100,000 Agents)")
    plt.xlabel("Day")
    plt.ylabel("Relative Scale (Normalized 0-1)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plots" / "extrapolation_100k_combined.png", dpi=200)
    plt.close()

    print("Success. Extrapolation comparison plot generated.")

if __name__ == "__main__":
    main()
