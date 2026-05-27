"""
Ordinal regression of tradeSeq trajectory cluster from per-spot LR output from CellNEST.

Inputs:
  --lr-matrix     CSV (rows = spots, columns = LR pairs, values = counts).
                  First column is the spot barcode index. May be gzipped.
  --spot-meta     CSV with at least:
                    - barcode index matching --lr-matrix
                    - a column with tradeseq cluster labels (default
                      'tradeseq_cluster'), values in {Low, Intermediate, High}.
                      Any other value (e.g. 'Other', NaN) is excluded.
  --outdir        Directory for output CSV and HTML files.

Optional:
  --tradeseq-col  Column name in --spot-meta holding the cluster labels
                  (default: tradeseq_cluster).
  --top-n         How many LR pairs to show in the bar chart (default: 20).
  --seed          Random seed for the 5-fold CV split (default: 42).
  --min-spots     Minimum number of usable spots required to fit (default: 30).

Example:
  python run_ordinal_regression.py \\
      --lr-matrix     /path/to/lr_count_matrix_filtered.csv.gz \\
      --spot-meta     /path/to/spot_meta_filtered.csv \\
      --outdir        ./ordinal_out \\
      --tradeseq-col  tradeseq_cluster

Notes:
  Both inputs are produced by the upstream spot-level pipeline, this script usese the saved per-condition outputs.
  The order Low -> Intermediate -> High is hardcoded, change STAGE_ORDER below if you need a different trajectory ordering.
"""

# from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Ascending trajectory order; index in this list IS the integer encoding.
# Low=0, Intermediate=1, High=2.
STAGE_ORDER: list[str] = ["Low", "Intermediate", "High"]
STAGE_INT: dict[str, int] = {s: i for i, s in enumerate(STAGE_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# Core regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_ordinal(
    lr_matrix: pd.DataFrame,
    spot_meta: pd.DataFrame,
    tradeseq_col: str,
    min_spots: int = 30,
) -> dict | None:
    """Fit a proportional-odds ordinal model.

    Returns a dict with the fitted statsmodels result, coefficient table,
    threshold table, training subset, and standardiser. Returns None if there
    are insufficient spots or if fitting fails.
    """
    # ── Restrict to spots with a valid trajectory label ──────────────────────
    joined = (
        lr_matrix.join(spot_meta[[tradeseq_col]], how="inner")
        .dropna(subset=[tradeseq_col])
    )
    joined = joined[joined[tradeseq_col].isin(STAGE_ORDER)]

    if len(joined) < min_spots:
        print(
            f"Only {len(joined)} usable spots "
            f"(min_spots={min_spots}); skipping fit."
        )
        return None

    features = [c for c in joined.columns if c != tradeseq_col]
    # .astype(float) defends against nullable Int64 columns which would
    # otherwise propagate object-dtype through OrderedModel's internal
    # np.asarray() and cause fitting to fail.
    X_raw = joined[features].values.astype(float)
    y_str = joined[tradeseq_col].values

    classes, counts = np.unique(y_str, return_counts=True)
    print(f"  n={len(joined):,}  features={len(features):,}")
    print(f"  class counts: {dict(zip(classes, counts))}")
    if len(classes) < 2:
        print("Only one class present; skipping fit.")
        return None

    # ── Standardise features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    X_df = pd.DataFrame(X_sc, columns=features).astype(float)

    # ── Integer-encode y (Low=0, Intermediate=1, High=2) ─────────────────────
    # Must use integer y rather than pd.Categorical: OrderedModel applies
    # np.asarray() internally which would cast Categorical to object dtype.
    y_int = pd.Series(
        [STAGE_INT[v] for v in y_str], name=tradeseq_col
    ).astype(int)

    # ── Fit full model ───────────────────────────────────────────────────────
    try:
        model = OrderedModel(y_int, X_df, distr="logit")
        result = model.fit(method="bfgs", disp=False)
    except np.linalg.LinAlgError as e:
        print(f"Singular matrix during fit: {e}")
        return None
    except Exception as e:
        print(f"Ordinal model fitting failed: {e}")
        return None

    print(result.summary())

    # ── Extract coefficients and thresholds ──────────────────────────────────
    n_thresh = len(STAGE_ORDER) - 1
    coef_vals = result.params.iloc[:-n_thresh]
    thresh_vals = result.params.iloc[-n_thresh:]
    coef_pvals = result.pvalues.iloc[:-n_thresh]
    thresh_se = result.bse.iloc[-n_thresh:]

    coef_df = pd.DataFrame(
        {
            "lr_pair": features,
            "coef": coef_vals.values,
            "pval": coef_pvals.values,
        }
    )
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df["sig"] = coef_df["pval"] < 0.05
    coef_df["direction"] = coef_df["coef"].apply(
        lambda v: "High (positive)" if v > 0 else "Low (negative)"
    )
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    thresh_df = pd.DataFrame(
        {
            "boundary": [
                f"{STAGE_ORDER[i]}/{STAGE_ORDER[i + 1]}" for i in range(n_thresh)
            ],
            "threshold": thresh_vals.values,
            "se": thresh_se.values,
        }
    )
    thresh_df["lo"] = thresh_df["threshold"] - thresh_df["se"]
    thresh_df["hi"] = thresh_df["threshold"] + thresh_df["se"]

    return {
        "result": result,
        "coef_df": coef_df,
        "thresh_df": thresh_df,
        "X_raw": X_raw,
        "y_str": y_str,
        "features": features,
        "scaler": scaler,
    }


def cross_validate_ordinal(
    X_raw: np.ndarray,
    y_str: np.ndarray,
    features: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[float]:
    """5-fold stratified CV refitting the ordinal model in each fold.

    The scaler is refit inside each training fold to avoid leakage.
    Returns the list of per-fold balanced accuracies. Folds that fail to
    converge are dropped from the list (the caller can detect this from len).
    """
    y_int = np.array([STAGE_INT[v] for v in y_str], dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_accs: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_int), start=1):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_raw[train_idx])
        X_te = scaler_fold.transform(X_raw[test_idx])

        y_tr_int = pd.Series(y_int[train_idx]).astype(int)
        y_te_str = y_str[test_idx]

        X_tr_df = pd.DataFrame(X_tr, columns=features).astype(float)
        X_te_df = pd.DataFrame(X_te, columns=features).astype(float)

        try:
            m = OrderedModel(y_tr_int, X_tr_df, distr="logit")
            r = m.fit(method="bfgs", disp=False)
            probs = r.predict(X_te_df)
            y_pred = [STAGE_ORDER[i] for i in probs.values.argmax(axis=1)]
            acc = balanced_accuracy_score(y_te_str, y_pred)
            cv_accs.append(acc)
            print(f"  fold {fold_i}: balanced accuracy = {acc:.3f}")
        except Exception as e:
            print(f"  fold {fold_i}: failed ({e.__class__.__name__}: {e})")

    return cv_accs


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_coefficients(coef_df: pd.DataFrame, top_n: int, title: str) -> alt.Chart:
    """Bar chart of the top-|coef| LR pairs."""
    top = coef_df.head(top_n).copy()
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            alt.X(
                "coef:Q",
                title="Coefficient  (positive → High,  negative → Low)",
            ),
            alt.Y(
                "lr_pair:N",
                sort=alt.EncodingSortField("abs_coef", order="descending"),
                axis=alt.Axis(labelFontSize=7),
                title="LR Pair",
            ),
            alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["High (positive)", "Low (negative)"],
                    range=["#d62728", "#1f77b4"],
                ),
                legend=alt.Legend(title="Direction"),
            ),
            alt.Opacity(
                "sig:N",
                scale=alt.Scale(domain=[True, False], range=[1.0, 0.35]),
                legend=alt.Legend(title="p<0.05"),
            ),
            tooltip=[
                "lr_pair",
                alt.Tooltip("coef:Q", format=".4f"),
                alt.Tooltip("pval:Q", format=".3e"),
                alt.Tooltip("abs_coef:Q", format=".4f"),
            ],
        )
        .properties(
            width=340,
            height=max(220, top_n * 14),
            title=alt.TitleParams(
                [
                    title,
                    f"Top {top_n} LR pairs by |coefficient|",
                    "Faded bars = p ≥ 0.05",
                ],
                fontSize=9,
            ),
        )
        .configure_view(strokeWidth=0)
    )
    return chart


def plot_thresholds(thresh_df: pd.DataFrame, title: str) -> alt.Chart:
    """Point + error-bar plot of the two ordinal cut-points."""
    pts = (
        alt.Chart(thresh_df)
        .mark_point(filled=True, size=80, color="steelblue")
        .encode(
            alt.X("threshold:Q", title="Threshold on latent scale"),
            alt.Y(
                "boundary:N",
                title="Class boundary",
                sort=alt.EncodingSortField("threshold", order="ascending"),
            ),
            tooltip=[
                "boundary",
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("se:Q", format=".3f"),
            ],
        )
    )
    err = (
        alt.Chart(thresh_df)
        .mark_rule(color="steelblue")
        .encode(alt.Y("boundary:N"), alt.X("lo:Q"), alt.X2("hi:Q"))
    )
    return (
        (pts + err)
        .properties(
            width=280,
            height=120,
            title=alt.TitleParams(
                [title, "Ordinal thresholds (±1 SE)"], fontSize=9
            ),
        )
        .configure_view(strokeWidth=0)
    )


def plot_cv(cv_accs: list[float], title: str) -> alt.Chart:
    """Per-fold bar chart with mean and chance reference lines."""
    mean_acc = float(np.mean(cv_accs))
    std_acc = float(np.std(cv_accs))
    cv_df = pd.DataFrame(
        {
            "fold": [f"Fold {i + 1}" for i in range(len(cv_accs))],
            "accuracy": cv_accs,
            "mean": [mean_acc] * len(cv_accs),
        }
    )
    bars = (
        alt.Chart(cv_df)
        .mark_bar(color="steelblue", opacity=0.8)
        .encode(
            alt.X("fold:N", title="Fold"),
            alt.Y(
                "accuracy:Q",
                title="Balanced Accuracy",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["fold", alt.Tooltip("accuracy:Q", format=".3f")],
        )
    )
    mean_rule = (
        alt.Chart(cv_df)
        .mark_rule(color="crimson", strokeDash=[4, 3], size=1.5)
        .encode(alt.Y("mean:Q"))
    )
    chance_rule = (
        alt.Chart(pd.DataFrame({"y": [1 / len(STAGE_ORDER)]}))
        .mark_rule(color="grey", strokeDash=[2, 2], size=1)
        .encode(alt.Y("y:Q"))
    )
    return (
        (bars + mean_rule + chance_rule)
        .properties(
            width=260,
            height=180,
            title=alt.TitleParams(
                [
                    title,
                    f"Ordinal CV: {mean_acc:.3f} ± {std_acc:.3f}  "
                    "(red = mean, grey = chance)",
                ],
                fontSize=9,
            ),
        )
        .configure_view(strokeWidth=0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# I/O and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(
    lr_path: Path, meta_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the LR count matrix and spot metadata, both indexed by barcode."""
    lr = pd.read_csv(lr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  LR matrix : {lr.shape[0]:,} spots × {lr.shape[1]:,} LR pairs")
    print(f"  spot meta : {meta.shape[0]:,} rows × {meta.shape[1]} cols")
    common = lr.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(
            "No common barcodes between --lr-matrix and --spot-meta indices. "
            "Check that both files use the same barcode format."
        )
    if len(common) < len(lr) or len(common) < len(meta):
        print(
            f"  intersect : {len(common):,} spots present in both "
            "(non-overlapping spots will be dropped at the join step)"
        )
    return lr.loc[common], meta.loc[common]


def drop_zero_variance_features(lr: pd.DataFrame) -> pd.DataFrame:
    """Drop LR pairs with zero variance across the supplied spots."""
    keep = lr.std() > 0
    dropped = (~keep).sum()
    if dropped:
        print(f"  dropping {dropped} zero-variance LR pairs")
    return lr.loc[:, keep]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lr-matrix", type=Path, required=True,
                   help="CSV of spot × LR-pair counts (optionally .csv.gz).")
    p.add_argument("--spot-meta", type=Path, required=True,
                   help="CSV of per-spot metadata indexed by barcode.")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Directory for outputs (created if missing).")
    p.add_argument("--tradeseq-col", default="tradeseq_cluster",
                   help="Column in --spot-meta with trajectory labels "
                        "(default: tradeseq_cluster).")
    p.add_argument("--prefix", default="ordinal_tradeseq",
                   help="Prefix for output filenames "
                        "(default: ordinal_tradeseq).")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top LR pairs shown in the coefficient bar chart.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the 5-fold CV split.")
    p.add_argument("--min-spots", type=int, default=30,
                   help="Minimum spots with a valid label required to fit.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip Altair plot generation; only write the CSV.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs:")
    print(f"  --lr-matrix : {args.lr_matrix}")
    print(f"  --spot-meta : {args.spot_meta}")
    lr, meta = load_inputs(args.lr_matrix, args.spot_meta)

    if args.tradeseq_col not in meta.columns:
        raise SystemExit(
            f"Column '{args.tradeseq_col}' not in --spot-meta. "
            f"Available columns: {list(meta.columns)}"
        )

    print("\nFiltering LR matrix for the labelled subset:")
    keep_idx = meta[meta[args.tradeseq_col].isin(STAGE_ORDER)].index
    lr_sub = drop_zero_variance_features(lr.loc[lr.index.intersection(keep_idx)])

    print("\nFitting ordinal regression:")
    fit = fit_ordinal(
        lr_sub, meta, tradeseq_col=args.tradeseq_col, min_spots=args.min_spots
    )
    if fit is None:
        print("\n❌  Fit did not complete; no outputs written.")
        return 1

    coef_path = args.outdir / f"{args.prefix}_coefs.csv"
    fit["coef_df"].to_csv(coef_path, index=False)
    print(f"\n✅  Wrote coefficients → {coef_path}")

    thresh_path = args.outdir / f"{args.prefix}_thresholds.csv"
    fit["thresh_df"].to_csv(thresh_path, index=False)
    print(f"✅  Wrote thresholds   → {thresh_path}")

    print("\n5-fold stratified CV:")
    cv_accs = cross_validate_ordinal(
        fit["X_raw"], fit["y_str"], fit["features"], seed=args.seed
    )
    cv_path = args.outdir / f"{args.prefix}_cv_accuracies.csv"
    pd.DataFrame(
        {"fold": [f"Fold {i + 1}" for i in range(len(cv_accs))], "accuracy": cv_accs}
    ).to_csv(cv_path, index=False)
    print(f"✅  Wrote CV accuracies → {cv_path}")

    if args.no_plots:
        print("\nSkipping plots (--no-plots).")
        return 0

    print("\nGenerating plots:")
    title = f"Ordinal regression — {args.prefix}"

    coef_chart = plot_coefficients(fit["coef_df"], args.top_n, title)
    coef_html = args.outdir / f"{args.prefix}_coefs.html"
    coef_chart.save(str(coef_html))
    print(f"✅  Wrote coefficient plot → {coef_html}")

    thresh_chart = plot_thresholds(fit["thresh_df"], title)
    thresh_html = args.outdir / f"{args.prefix}_thresholds.html"
    thresh_chart.save(str(thresh_html))
    print(f"✅  Wrote threshold plot  → {thresh_html}")

    if cv_accs:
        cv_chart = plot_cv(cv_accs, title)
        cv_html = args.outdir / f"{args.prefix}_cv.html"
        cv_chart.save(str(cv_html))
        print(f"✅  Wrote CV plot         → {cv_html}")
    else:
        print("⚠️  All CV folds failed; no CV plot written.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())