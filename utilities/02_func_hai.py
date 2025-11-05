# ===============================
# Imports & Global Config
# ===============================
import os, json, math, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140
sns.set_context("notebook")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Paths
DATA_PATH  = "data/housing_adequacy_dataset.csv"  # must contain: quarter, province, hai, dwelling_starts, pop_change_q, needed_units_q


# Holdout split
CUTOFF_DATE = "2018-12-31"  # train ≤ cutoff, test > cutoff

# Models to try
ALL_MODELS = ["lr", "rf", "ridge", "xgb", "etr"]

# ===============================
# Causal smoothing helper
# ===============================
def causal_roll(series: pd.Series, window: int, how: str = "median") -> pd.Series:
    """One-sided rolling aggregator (no look-ahead)."""
    if how == "median":
        return series.rolling(window=window, min_periods=1).median()
    elif how == "mean":
        return series.rolling(window=window, min_periods=1).mean()
    else:
        raise ValueError("how must be 'median' or 'mean'")

# ===============================
# Build modeling frame (HAI)
# ===============================
def build_model_frame_hai(
    df: pd.DataFrame,
    *,
    smooth_window: int = 4,
    smooth_func: str = "median",
    use_smooth_features: bool = False,
    predict_smooth_target: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      model_df : dataframe with y, lags (raw + optional smoothed), exog lags, seasonal dummies
      meta     : dict describing chosen smoothing flags to encode cache names downstream
    """
    df = df.sort_values(["province", "quarter"]).copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()

    # base series
    df["hai_raw"] = df["hai"].astype(float)

    # smoothed (causal)
    df["hai_roll"] = (
        df.groupby("province")["hai_raw"]
          .transform(lambda s: causal_roll(s, smooth_window, smooth_func))
    )

    # supervised target
    df["y"] = df["hai_roll"] if predict_smooth_target else df["hai_raw"]

    # y lags (raw & smooth)
    for L in [1, 2, 3, 4, 8]:
        df[f"hai_raw_lag{L}"]  = df.groupby("province")["hai_raw"].shift(L)
        df[f"hai_roll_lag{L}"] = df.groupby("province")["hai_roll"].shift(L)

    # exogenous lags
    for L in [1, 4, 8]:
        for col in ["dwelling_starts", "pop_change_q", "needed_units_q"]:
            df[f"{col}_lag{L}"] = df.groupby("province")[col].shift(L)

    # seasonality dummies
    qdum = pd.get_dummies(df["quarter"].dt.quarter, prefix="q")
    if "q_1" in qdum:
        qdum = qdum.drop(columns=["q_1"])
    df = pd.concat([df, qdum], axis=1)

    meta = dict(
        smooth_window=smooth_window,
        smooth_func=smooth_func,
        use_smooth_features=use_smooth_features,
        predict_smooth_target=predict_smooth_target
    )
    return df, meta

# ===============================
# Feature selection by horizon
# ===============================
def features_for_choice_hai(choice: str, *, use_smooth_features: bool):
    choice = choice.lower()
    if choice == "next_quarter":
        H = 1
        feats = [
            "hai_raw_lag1", "hai_raw_lag4",  # memory + seasonality
            "dwelling_starts_lag1", "pop_change_q_lag1", "needed_units_q_lag1",
        ]
        if use_smooth_features:
            feats += ["hai_roll_lag1", "hai_roll_lag4"]
    elif choice == "same_quarter_next_year":
        H = 4
        feats = [
            "hai_raw_lag4", "hai_raw_lag8",
            "dwelling_starts_lag4", "pop_change_q_lag4", "needed_units_q_lag4",
        ]
        if use_smooth_features:
            feats += ["hai_roll_lag4", "hai_roll_lag8"]
    else:
        raise ValueError("choice must be 'next_quarter' or 'same_quarter_next_year'")

    # optionally add: feats += ["q_2","q_3","q_4"]
    return H, feats

# ===============================
# Splits & metrics
# ===============================
def chrono_split(df, cutoff):
    cutoff = pd.Timestamp(cutoff)
    tr = df[df["quarter"] <= cutoff].copy()
    te = df[df["quarter"] >  cutoff].copy()
    return tr, te

def rolling_split(df, initial=None, step=1, fh=1):
    df = df.copy()
    df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
    dates = pd.Index(df["quarter"].dropna().unique()).sort_values()
    if len(dates) == 0:
        return
    if initial is None:
        initial = dates[int(0.6 * len(dates))]
    else:
        initial = pd.Timestamp(initial)
    start_idx = dates.get_indexer([initial], method="ffill")[0]
    start_idx = max(start_idx, 0)
    for i in range(start_idx, len(dates) - fh, step):
        train_end = dates[i]
        test_slice = dates[i+1 : i+1+fh]
        tr = df[df["quarter"] <= train_end].copy()
        te = df[df["quarter"].isin(test_slice)].copy()
        if not te.empty:
            yield tr, te

def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = 1e-8
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}

def _naive_mae_train(y_train: pd.Series, season: int = 1, eps: float = 1e-12) -> float:
    y = y_train.to_numpy()
    if len(y) <= season:
        return np.nan
    denom = np.mean(np.abs(y[season:] - y[:-season]))
    return max(denom, eps)

# ===============================
# Models & param grids
# ===============================
def make_model(name, **kw):
    if name == "lr":
        model = LinearRegression()
    elif name == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("est", Ridge())])
    elif name == "lasso":
        model = Pipeline([("scaler", StandardScaler()), ("est", Lasso(max_iter=20000))])
    elif name == "enet":
        model = Pipeline([("scaler", StandardScaler()), ("est", ElasticNet(max_iter=20000))])
    elif name == "rf":
        model = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
    elif name == "etr":
        model = ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE)
    elif name == "gbr":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    elif name == "svr":
        model = Pipeline([("scaler", StandardScaler()), ("est", SVR())])
    elif name == "xgb":
        model = XGBRegressor(tree_method="hist", n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")
    if kw:
        model.set_params(**kw)
    return model

PARAM_SPACE = {
    "lr":   {},
    "ridge": {"est__alpha": np.logspace(-2, 2, 9)},
    "lasso": {"est__alpha": np.logspace(-3, 0.5, 8)},
    "enet":  {"est__alpha": np.logspace(-3, 0.5, 6), "est__l1_ratio": [0.2, 0.5, 0.8]},
    "rf":   {"n_estimators": [200, 400], "max_depth": [None, 8], "min_samples_split": [2, 5],
             "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.8]},
    "etr":  {"n_estimators": [400, 800], "max_depth": [None, 10], "min_samples_split": [2, 5],
             "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.8]},
    "gbr":  {"n_estimators": [300, 600], "learning_rate": [0.03, 0.1], "max_depth": [2, 3], "subsample": [0.85, 1.0]},
    "svr":  {"est__C": [0.3, 1, 3, 10], "est__epsilon": [0.01, 0.1, 0.3], "est__gamma": ["scale"]},
    "xgb":  {"max_depth": [3, 6], "min_child_weight": [1, 4], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
}

# ===============================
# Tuning + cache
# ===============================
def sample_param_grid(space: dict, n_iter: int, rng: np.random.RandomState):
    if not space:
        return [dict()]
    keys = list(space.keys())
    samples = []
    for _ in range(n_iter):
        d = {}
        for k in keys:
            v = space[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                d[k] = v[rng.randint(0, len(v))]
            else:
                d[k] = v
        samples.append(d)
    return samples

def rolling_score_one_fold(model, tr, te, features, season=1):
    Xtr, ytr = tr[features], tr["y"]
    Xte, yte = te[features], te["y"]
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    denom = _naive_mae_train(ytr, season=season)
    mae = mean_absolute_error(yte, ypred)
    return mae / denom

def tune_model_for_province(df, features, model_name, n_iter=20, fh=1, initial=None,
                            season=1, random_state=RANDOM_STATE):
    space = PARAM_SPACE[model_name]
    rng = np.random.RandomState(random_state)
    candidates = sample_param_grid(space, n_iter=n_iter, rng=rng)

    scores = []
    for params in candidates:
        fold_scores = []
        for tr, te in rolling_split(df, initial=initial, fh=fh):
            m = make_model(model_name, **params)
            try:
                s = rolling_score_one_fold(m, tr, te, features, season=season)
                if np.isfinite(s):
                    fold_scores.append(s)
            except Exception:
                pass
        score = np.mean(fold_scores) if len(fold_scores) else np.inf
        scores.append((params, score))
    scores.sort(key=lambda x: x[1])
    best_params, best_score = scores[0]
    return best_params, best_score, scores

def tune_all_models_per_province(train_df, features, models_to_run, n_iter=20, fh=1,
                                 initial=None, season=1, random_state=RANDOM_STATE):
    best_params = {}
    provinces = list(train_df["province"].unique())
    total_tasks = len(provinces) * len(models_to_run)
    task = 0
    print(f"Tuning: {len(provinces)} prov × {len(models_to_run)} models × {n_iter} trials (fh={fh}, season={season})")
    for prov, gtr in train_df.groupby("province"):
        for name in models_to_run:
            task += 1
            print(f"→ [{task}/{total_tasks}] {prov.upper()} — {name.upper()} ...", end=" ", flush=True)
            bp, bs, _ = tune_model_for_province(
                df=gtr, features=features, model_name=name, n_iter=n_iter, fh=fh,
                initial=initial, season=season, random_state=random_state
            )
            best_params[(prov, name)] = bp
            print(f"best MASE={bs:.3f}")
    print("Tuning done.")
    return best_params

def cache_path_for_hai(choice: str, meta: dict) -> str:
    tag = "h1" if choice == "next_quarter" else "h4"
    smooth_tag = f"s{meta['smooth_window']}_{meta['smooth_func']}"
    feat_tag = "featS" if meta["use_smooth_features"] else "featN"
    targ_tag = "tSm" if meta["predict_smooth_target"] else "tRaw"
    # NOTE: this naming matches your prior scheme; new flag combos create NEW files.
    return f"best_params_cache_hai_{tag}_{smooth_tag}_{feat_tag}_{targ_tag}.json"

def load_or_tune_best_params(train_df, features, models_to_run,
                             prediction_choice: str, meta: dict,
                             n_iter=8, fh=1, initial=None):
    cache_path = cache_path_for_hai(prediction_choice, meta)
    if os.path.exists(cache_path):
        print(f"Loading cached best parameters from {cache_path}")
        with open(cache_path, "r") as f:
            best_params = json.load(f)
        best_params = {tuple(k.split("|")): v for k, v in best_params.items()}
        return best_params, cache_path
    else:
        print("Running tuning from scratch...")
        season = 1 if prediction_choice == "next_quarter" else 4
        best_params = tune_all_models_per_province(
            train_df=train_df,
            features=features,
            models_to_run=models_to_run,
            n_iter=n_iter,
            fh=fh,
            initial=initial,
            season=season,
            random_state=RANDOM_STATE
        )
        best_params_str = {"|".join(k): v for k, v in best_params.items()}
        with open(cache_path, "w") as f:
            json.dump(best_params_str, f, indent=2)
        print(f"Saved tuned best parameters to {cache_path}")
        return best_params, cache_path

# ===============================
# Holdout fit/predict + overlay
# ===============================
def fit_predict_holdout_per_province(train_df, test_df, features, models_to_run, best_params,
                                     prediction_choice: str):
    season = 1 if prediction_choice == "next_quarter" else 4
    rows = []
    for prov, gtr in train_df.groupby("province"):
        gte = test_df[test_df["province"] == prov]
        if gte.empty:
            continue
        for name in models_to_run:
            params = best_params.get((prov, name), {})
            m = make_model(name, **params)
            Xtr, ytr = gtr[features], gtr["y"]
            Xte, yte = gte[features], gte["y"]
            m.fit(Xtr, ytr)
            ypred = m.predict(Xte)

            res = metrics(yte, ypred)
            denom = _naive_mae_train(ytr, season=season)
            res["MASE"] = res["MAE"] / denom

            for q, yt, yp in zip(gte["quarter"].values, yte.values, ypred):
                rows.append({
                    "province": prov, "model": name,
                    "quarter": pd.to_datetime(q), "y_true": yt, "y_pred": yp,
                    **res
                })
    return pd.DataFrame(rows)

def summarize_metrics_table(preds_df: pd.DataFrame):
    g_model = (preds_df.groupby("model")[["MAE","RMSE","sMAPE","MASE"]]
               .mean().round(3).sort_values("MAE"))
    display(g_model)
    g_pm = (preds_df.groupby(["province","model"])[["MAE","RMSE","sMAPE","MASE"]]
            .mean().round(3))
    return g_model, g_pm

def plot_holdout_overlay(preds_df, models_to_run, suptitle="HAI — True vs Predicted (Holdout)"):
    provs = sorted(preds_df["province"].unique())
    cols = 4
    rows = math.ceil(len(provs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.6*rows), sharex=False, sharey=False)
    axes = axes.ravel()

    for i, prov in enumerate(provs):
        ax = axes[i]
        g = preds_df[preds_df["province"] == prov].copy()
        truth = g.drop_duplicates("quarter")[["quarter","y_true"]].sort_values("quarter")
        ax.plot(truth["quarter"], truth["y_true"], color="black", lw=1.8, alpha=.55, label="True")

        for name in models_to_run:
            gm = g[g["model"] == name].sort_values("quarter")
            if gm.empty: continue
            ax.plot(gm["quarter"], gm["y_pred"], "--", lw=1.5, label=name.upper())

        ax.set_title(prov.upper(), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(i+1, rows*cols):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(2+len(models_to_run), 6),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

# ===============================
# Rolling overlays (with MASE denom)
# ===============================
def rolling_evolution_overlay(df_all, features, best_params, models_to_plot,
                              fh=1, initial=None, show_history=False, step=1, last_k_cutoffs=None,
                              return_df=False):
    models_to_plot = [m.lower() for m in models_to_plot]
    rows = []
    folds = list(rolling_split(df_all, initial=initial, fh=fh, step=step))
    if last_k_cutoffs is not None and len(folds) > last_k_cutoffs:
        folds = folds[-last_k_cutoffs:]

    for tr, te in folds:
        cutoff = pd.to_datetime(tr["quarter"].max())
        for prov, gtr in tr.groupby("province"):
            gte = te[te["province"] == prov]
            if gte.empty: continue
            gtr_ = gtr.dropna(subset=features + ["y"])
            gte_ = gte.dropna(subset=features + ["y"])
            if gtr_.empty or gte_.empty: continue
            Xtr, ytr = gtr_[features], gtr_["y"]
            Xte, yte = gte_[features], gte_["y"]

            # per-fold seasonal-naive MAE denom for MASE (season == fh)
            mase_denom = float(np.mean(np.abs(ytr.diff(fh).dropna()))) if len(ytr) > fh else np.nan

            for name in models_to_plot:
                params = best_params.get((prov, name), {})
                try:
                    m = make_model(name, **params)
                    m.fit(Xtr, ytr)
                    yhat = m.predict(Xte)
                    if not np.isfinite(yhat).any(): continue
                    rows.append(pd.DataFrame({
                        "cutoff": cutoff, "province": prov, "model": name,
                        "quarter": pd.to_datetime(gte_["quarter"].values),
                        "y_true": yte.values, "y_pred": yhat,
                        "MASE_denom": mase_denom
                    }))
                except Exception:
                    continue

    if not rows:
        print("No rolling predictions generated.")
        return None if return_df else None

    df = pd.concat(rows, ignore_index=True).sort_values(["province","model","cutoff","quarter"])
    cuts = sorted(df["cutoff"].unique())
    cut_rank = {c:i for i,c in enumerate(cuts)}
    df["cut_rank"] = df["cutoff"].map(cut_rank)

    # Plot latest-only overlay
    provs = sorted(df["province"].unique())
    cols = 4
    rows_n = math.ceil(len(provs)/cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4*cols, 2.6*rows_n), sharex=False, sharey=False)
    axes = axes.ravel()
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or ["C0","C1","C2","C3","C4","C5"]
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(models_to_plot)}

    for i, prov in enumerate(provs):
        ax = axes[i]
        g = df[df["province"] == prov]
        truth = g.drop_duplicates("quarter")[["quarter","y_true"]].sort_values("quarter")
        ax.plot(truth["quarter"], truth["y_true"], color="black", lw=1.4, alpha=.45, label="True")
        for m in models_to_plot:
            gm = g[g["model"] == m]
            if gm.empty: continue
            c = model_colors[m]
            if show_history:
                for cu, gc in gm.groupby("cutoff"):
                    ax.plot(gc["quarter"], gc["y_pred"], "--", color=c, alpha=0.25 + 0.6*cut_rank[cu]/max(len(cuts)-1,1), lw=1.0)
            latest_cut = gm["cutoff"].max()
            gl = gm[gm["cutoff"] == latest_cut].sort_values("quarter")
            if not gl.empty:
                ax.plot(gl["quarter"], gl["y_pred"], "--", color=c, lw=1.8, label=m.upper())
        ax.set_title(prov.upper(), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(i+1, rows_n*cols):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(2+len(models_to_plot), 6),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("HAI — Rolling forecast overlay (latest cutoff)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

    if return_df:
        return df

def latest_per_quarter(df_preds):
    return (df_preds.sort_values(["province","model","quarter","cutoff"])
                   .drop_duplicates(["province","model","quarter"], keep="last")
                   .sort_values(["province","model","quarter"]))

def rolling_overlay_lines(df_preds, models_to_plot):
    models_to_plot = [m.lower() for m in models_to_plot]
    df = df_preds.copy()
    df["model"] = df["model"].str.lower()

    provs = sorted(df["province"].unique())
    cols = 4
    rows_n = math.ceil(len(provs)/cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4*cols, 2.6*rows_n), sharex=False, sharey=False)
    axes = axes.ravel()

    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3","C4","C5"])
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(models_to_plot)}
    model_styles = {m: s for m, s in zip(models_to_plot, ["--","-.",":",(0,(3,1,1,1)),(0,(5,1))])}

    for i, prov in enumerate(provs):
        ax = axes[i]
        g = df[df["province"] == prov]
        truth = g.drop_duplicates("quarter")[["quarter","y_true"]].sort_values("quarter")
        ax.plot(truth["quarter"], truth["y_true"], color="black", lw=1.3, alpha=.45, label="True")
        for m in models_to_plot:
            gl = g[g["model"] == m].sort_values("quarter")
            if gl.empty: continue
            ax.plot(gl["quarter"], gl["y_pred"],
                    linestyle=model_styles[m], color=model_colors[m],
                    marker="o", markersize=2.2, lw=1.5, label=m.upper())
        ax.set_title(prov.upper(), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, ls="--", alpha=.3)

    for j in range(i+1, rows_n*cols):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(2+len(models_to_plot), 6), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("HAI — Rolling forecast (stitched latest per quarter)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

# ===============================
# Heatmaps, ranking & helpers
# ===============================
def add_row_metrics_for_rolling(df, mase_denom_col="MASE_denom", eps=1e-8):
    d = df.copy()
    err = d["y_pred"] - d["y_true"]
    d["MAE"]   = err.abs()
    d["SE"]    = err**2
    d["sMAPE"] = 100 * (2*err.abs() / (d["y_true"].abs() + d["y_pred"].abs() + eps))
    if mase_denom_col in d.columns:
        denom = d[mase_denom_col].replace(0, np.nan)
        d["MASE"] = d["MAE"] / denom
    return d

def plot_perf_heatmaps_from_preds(
    preds_df: pd.DataFrame,
    metrics=("MASE", "MAE", "RMSE", "sMAPE"),
    *,
    center_on_1=("MASE",),
    aggfunc="mean",
    annotate=True,
    fmt=".2f",
    cmap_div="coolwarm",
    cmap_seq="viridis",
    figsize_per_panel=(6.0, 5.0),
    ncols=2,
    sort_by="MASE",
    mase_season_denom_by_province: dict | None = None,
    mase_denom_col: str | None = None,
    mase_fallback_to_existing_col: bool = True,
    smape_scale: float = 100.0,
    eps: float = 1e-8,
):
    req = {"province", "model", "y_true", "y_pred"}
    missing = req - set(preds_df.columns)
    if missing:
        raise ValueError(f"preds_df must contain {req}; missing: {missing}")

    aggfunc_callable = {"mean": np.mean, "median": np.median}[aggfunc] if isinstance(aggfunc, str) else aggfunc
    df = preds_df.copy()
    df["model"] = df["model"].str.lower()

    def _per_group_metrics(g: pd.DataFrame) -> dict:
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        err = yp - yt
        out = {}
        for m in metrics:
            key = m.upper()
            if key == "MAE":
                out["MAE"] = np.mean(np.abs(err))
            elif key == "RMSE":
                out["RMSE"] = np.sqrt(np.mean(err**2))
            elif key == "SMAPE":
                denom = (np.abs(yt) + np.abs(yp) + eps)
                out["sMAPE"] = smape_scale * np.mean(2.0 * np.abs(err) / denom)
            elif key == "MASE":
                if mase_season_denom_by_province is not None:
                    prov = g["province"].iloc[0]
                    if prov in mase_season_denom_by_province and mase_season_denom_by_province[prov] not in (None, 0):
                        denom = float(mase_season_denom_by_province[prov])
                        out["MASE"] = np.mean(np.abs(err)) / denom
                        continue
                if mase_denom_col is not None and mase_denom_col in g.columns:
                    denom_vals = g[mase_denom_col].dropna().to_numpy()
                    if denom_vals.size > 0 and np.all(denom_vals > 0):
                        denom = float(np.mean(denom_vals))
                        out["MASE"] = np.mean(np.abs(err)) / denom
                        continue
                if mase_fallback_to_existing_col and "MASE" in g.columns:
                    out["MASE"] = aggfunc_callable(g["MASE"].to_numpy())
        return out

    per_g = (df.groupby(["province", "model"], as_index=False)
               .apply(lambda g: pd.Series(_per_group_metrics(g)))
               .reset_index(drop=True))

    if sort_by not in per_g.columns:
        if sort_by in df.columns:
            per_g_sort = (df.groupby(["province", "model"], as_index=False)
                            .agg(**{sort_by: (sort_by, aggfunc_callable)}))
            per_g = per_g.merge(per_g_sort, on=["province","model"], how="left")
        else:
            raise ValueError(f"Cannot sort by '{sort_by}' — metric not computable and not found in preds_df.")

    order_by_prov = (per_g.groupby("province")[sort_by]
                          .mean()
                          .sort_values(ascending=True))
    province_order = order_by_prov.index.tolist()
    model_order = sorted(per_g["model"].unique().tolist())

    m = len(metrics)
    ncols = max(1, ncols)
    nrows = math.ceil(m / ncols)
    fw, fh = figsize_per_panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw*ncols, fh*nrows), squeeze=False)
    sns.set_style("whitegrid")

    for i, metric in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        mkey = metric if metric in per_g.columns else metric.upper()
        if mkey not in per_g.columns:
            ax.set_visible(False)
            ax.set_title(f"{metric} (not available)", fontsize=10)
            continue

        pivot = (per_g.pivot(index="province", columns="model", values=mkey)
                       .reindex(index=province_order, columns=model_order))

        if metric.upper() in [x.upper() for x in center_on_1]:
            vmin_val = np.nanmin(pivot.values)
            vmax_val = np.nanmax(pivot.values)
            if np.isfinite(vmin_val) and np.isfinite(vmax_val):
                span = max(1 - vmin_val, vmax_val - 1)
                vmin, vmax = 1 - span, 1 + span
            else:
                vmin, vmax = None, None
            sns.heatmap(pivot, ax=ax, annot=annotate, fmt=fmt, cmap=cmap_div,
                        vmin=vmin, vmax=vmax, center=1.0, cbar_kws={"label": metric})
        else:
            sns.heatmap(pivot, ax=ax, annot=annotate, fmt=fmt, cmap=cmap_seq,
                        cbar_kws={"label": metric})

        ax.set_yticklabels([lab.get_text().upper() for lab in ax.get_yticklabels()], rotation=0)
        ax.set_xlabel("MODEL")
        ax.set_ylabel("PROVINCE")
        ax.set_title(f"{metric} by Province × Model ({aggfunc})", fontsize=11)

    total_axes = nrows * ncols
    for j in range(m, total_axes):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    plt.show()

def rank_models_across_provinces(
    preds_df: pd.DataFrame,
    *,
    metric: str = "MASE",    # lower is better
    agg: str = "mean",       # "mean" or "median"
    provinces: list[str] | None = None,
    models: list[str] | None = None,
    min_obs: int = 1,
    tie_breakers: list[str] | None = None,
    rank_method: str = "dense",
    return_tables: bool = True
):
    req_cols = {"province", "model", metric}
    missing = req_cols - set(preds_df.columns)
    if missing:
        raise ValueError(f"preds_df missing required columns: {missing}")

    df = preds_df.copy()
    df["model"] = df["model"].str.lower()
    if provinces is not None:
        df = df[df["province"].isin(provinces)]
    if models is not None:
        models = [m.lower() for m in models]
        df = df[df["model"].isin(models)]

    aggfunc = np.mean if agg == "mean" else np.median
    grouped = (df.groupby(["province", "model"], as_index=False)
                 .agg(metric_agg=(metric, aggfunc), n=(metric, "size")))
    grouped = grouped[grouped["n"] >= min_obs]
    if grouped.empty:
        raise ValueError("No (province, model) groups remain after filtering.")

    if tie_breakers:
        tb = (df.groupby(["province", "model"], as_index=False)
                .agg(**{f"{m}_agg": (m, aggfunc) for m in tie_breakers if m in df.columns}))
        grouped = grouped.merge(tb, on=["province", "model"], how="left")

    def _rank_one(g):
        sort_cols = ["metric_agg"] + [f"{m}_agg" for m in (tie_breakers or []) if f"{m}_agg" in g.columns]
        g = g.sort_values(sort_cols, ascending=True, kind="mergesort")
        g["rank"] = g["metric_agg"].rank(method=rank_method, ascending=True)
        return g

    per_province_rank = (grouped.groupby("province", group_keys=False)
                                .apply(_rank_one)
                                .sort_values(["province", "rank", "metric_agg"])
                                .reset_index(drop=True))
    best_by_province = (per_province_rank[per_province_rank["rank"] == 1]
                        .sort_values(["metric_agg", "province"])
                        .reset_index(drop=True))
    leader = (per_province_rank.groupby("model")
              .agg(
                  wins=("rank", lambda s: int((s == 1).sum())),
                  avg_rank=("rank", "mean"),
                  med_rank=("rank", "median"),
                  mean_metric=("metric_agg", "mean"),
                  n_provinces=("province", "nunique")
              ).sort_values(["wins","avg_rank","mean_metric"], ascending=[False, True, True])
              .reset_index())
    if return_tables:
        per_province_rank["province"] = per_province_rank["province"].str.upper()
        best_by_province["province"]   = best_by_province["province"].str.upper()
        leader["model"] = leader["model"].str.upper()
        return per_province_rank, leader, best_by_province
    else:
        return None



# ===============================
# Run both input types (RAW & SMOOTHED)
# ===============================

def run_hai_experiment(
    *,
    prediction_choice: str,
    smooth_window: int,
    smooth_func: str,
    use_smooth_features: bool,
    predict_smooth_target: bool,
    models=ALL_MODELS,
    cutoff=CUTOFF_DATE,
    initial_roll="2012-12-31",
    roll_fh=None,       # if None: uses H from features_for_choice_hai
    roll_step=2,
    roll_last_k=8,
    n_iter_tune=8
):
    # 1) Load & build frame with the chosen smoothing flags
    raw = pd.read_csv(DATA_PATH)
    model_df, meta = build_model_frame_hai(
        raw,
        smooth_window=smooth_window,
        smooth_func=smooth_func,
        use_smooth_features=use_smooth_features,
        predict_smooth_target=predict_smooth_target
    )

    # 2) Horizon + features
    H, feat_cols = features_for_choice_hai(prediction_choice, use_smooth_features=use_smooth_features)
    season = 1 if prediction_choice == "next_quarter" else 4
    ROLL_FH = H if roll_fh is None else roll_fh
    print(f"\n=== CONFIG ===")
    print(f"Mode={prediction_choice} | H={H} | season={season}")
    print(f"Smooth: window={smooth_window}, func={smooth_func}, featS={use_smooth_features}, targetSm={predict_smooth_target}")
    print("Features:", feat_cols)

    # 3) Drop NaNs for selected features/target
    safe_df = model_df.dropna(subset=feat_cols + ["y"]).copy()

    # 4) Holdout split
    train, test = chrono_split(safe_df, cutoff=cutoff)
    print(f"Holdout: Train {train['quarter'].min().date()} → {train['quarter'].max().date()} | "
          f"Test {test['quarter'].min().date()} → {test['quarter'].max().date()}")

    # 5) Tune/load params (cache name encodes horizon + smoothing flags)
    best_params, cache_path = load_or_tune_best_params(
        train_df=train,
        features=feat_cols,
        models_to_run=models,
        prediction_choice=prediction_choice,
        meta=meta,
        n_iter=n_iter_tune,
        fh=H,
        initial=None
    )

    # 6) Holdout predictions + summary
    preds_hold = fit_predict_holdout_per_province(
        train_df=train, test_df=test, features=feat_cols,
        models_to_run=models, best_params=best_params,
        prediction_choice=prediction_choice
    )
    print("\n=== HAI — Holdout averages across provinces (tuned) ===")
    g_model, g_pm = summarize_metrics_table(preds_hold)
    plot_holdout_overlay(preds_hold, models, suptitle="HAI — Holdout overlay (tuned)")

    # 7) Rolling predictions + overlays
    preds_roll = rolling_evolution_overlay(
        df_all=safe_df,
        features=feat_cols,
        best_params=best_params,
        models_to_plot=models,
        fh=ROLL_FH,
        initial=initial_roll,
        step=roll_step,
        last_k_cutoffs=roll_last_k,
        show_history=False,
        return_df=True,
    )

    if preds_roll is not None:
        # stitched latest lines
        d_latest = latest_per_quarter(preds_roll)
        rolling_overlay_lines(d_latest, models)

    # 8) Heatmaps + Ranking
    print("\n=== Heatmaps & Rankings (Holdout) ===")
    plot_perf_heatmaps_from_preds(
        preds_df=preds_hold,
        metrics=("MASE","MAE","RMSE","sMAPE"),
        center_on_1=("MASE",),
        mase_fallback_to_existing_col=True
    )
    per_rank_h, leader_h, best_by_prov_h = rank_models_across_provinces(
        preds_df=preds_hold, metric="MASE", agg="mean", rank_method="dense"
    )
    print("Leaderboard (holdout by MASE):"); display(leader_h)

    if preds_roll is not None:
        print("\n=== Heatmaps & Rankings (Rolling) ===")
        roll_scored = add_row_metrics_for_rolling(preds_roll, mase_denom_col="MASE_denom")
        plot_perf_heatmaps_from_preds(
            preds_df=roll_scored,
            metrics=("MASE","MAE","RMSE","sMAPE"),
            center_on_1=("MASE",),
            mase_denom_col="MASE_denom"
        )
        per_rank_r, leader_r, best_by_prov_r = rank_models_across_provinces(
            preds_df=roll_scored, metric="MASE", agg="mean", rank_method="dense"
        )
        print("Leaderboard (rolling by MASE):"); display(leader_r)
    else:
        roll_scored = None
        per_rank_r = leader_r = best_by_prov_r = None

    return {
        "cache_path": cache_path,
        "preds_holdout": preds_hold,
        "preds_rolling": preds_roll,
        "roll_scored": roll_scored,
        "leader_holdout": leader_h,
        "leader_rolling": leader_r
    }














