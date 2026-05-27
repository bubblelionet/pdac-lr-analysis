"""
Build the spot x LR-pair count matrix and per-spot metadata used by
run_ordinal_regression.py (and by the wider spot-level analysis pipeline).

For each sample, the script reads a CellNEST output CSV
(NEST_PDAC_<sample>_manualDB_thresholded.csv by default), builds an
LR-pair name from the ligand and receptor columns, prefixes spot barcodes
with the sample ID, and accumulates per-spot interaction counts. Counts are
incremented at BOTH endpoints (sending and receiving spot) for each
interaction. Spots that never appear in any retained interaction are not represented in the matrix.

The script then joins per-spot metadata from an AnnData file (.h5ad),
computes per-spot summary statistics (total LR count, LR-pair richness,
Hill number of order 1), and assigns each spot a tumour stage
(Early / Intermediate / Late / Non-tumour) based on the coarse pathologist
annotation and the predicted non-glandular probability.

Outputs per condition (a "condition" is a min-count threshold on column
sums of the matrix):

  <outdir>/<condition>/lr_count_matrix_<condition>.csv.gz
  <outdir>/<condition>/spot_meta_<condition>.csv

The "unfiltered" condition (min-count = 0) keeps every LR pair seen in any
sample; the "filtered" condition (default min-count = 50) keeps only LR
pairs with at least that many total counts across all spots.

Example:
  python build_spot_lr_inputs.py \\
      --nest-dir  /path/to/CellNEST_outputs \\
      --adata     /path/to/institutional_samples_untreated.h5ad \\
      --outdir    /path/to/spot_level_analysis \\
      --samples   exp1_B1 exp1_C1 exp1_D1 exp2_A1 exp2_B1 exp2_D1 \\
                  exp4_A1 exp4_C1 exp4_D1 exp6_A1 exp6_B1 exp6_D1 \\
      --conditions unfiltered=0 filtered=50

Required AnnData columns (configurable; defaults match the upstream pipeline):
  - prefixed_barcode  (spot barcode prefixed with "<exp>_<slide>_")
  - sample            (sample identifier)
  - tradeseq_cluster  (trajectory label: Low / Intermediate / High / Other)
  - Non-glandular probability  (continuous; used for tumour staging)
  - coarse            (coarse pathologist label; only 'Tumour' gets a stage)

"""

# from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Per-spot summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def hill_q1_per_spot(count_matrix: pd.DataFrame) -> pd.Series:
    """Hill number of order 1 (= exp Shannon entropy) per spot.

    Spots with zero total counts get hill_q1 = 0 rather than NaN.
    """
    totals = count_matrix.sum(axis=1)
    props = count_matrix.div(totals.replace(0, np.nan), axis=0)
    log_props = props.apply(lambda x: np.log(x.where(x > 0)))
    shannon = -(props * log_props).sum(axis=1)
    hill = np.exp(shannon)
    hill[totals == 0] = 0
    return hill


# ─────────────────────────────────────────────────────────────────────────────
# Build the spot × LR matrix from CellNEST CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_matrix(
    samples: list[str], nest_dir: Path, file_pattern: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read CellNEST CSVs for the given samples and return a sparse-like
    spot × LR-pair count matrix (dense pandas DataFrame, int dtype) and a
    barcode → sample lookup.

    For each interaction row, the LR-pair count is incremented at both
    the sending and the receiving spot.
    """
    long_records: list[pd.DataFrame] = []
    bc_to_sample: dict[str, str] = {}

    for sample in samples:
        path = nest_dir / file_pattern.format(sample=sample)
        if not path.exists():
            print(f"{sample}: NEST file not found at {path}, skipping.")
            continue

        df = pd.read_csv(path)
        required = {"from_cell", "to_cell", "ligand", "receptor"}
        missing = required - set(df.columns)
        if missing:
            print(f"{sample}: missing columns {sorted(missing)}, skipping.")
            continue

        df["lr_pair"] = df["ligand"].astype(str) + "-" + df["receptor"].astype(str)
        df["from_cell"] = f"{sample}_" + df["from_cell"].astype(str)
        df["to_cell"] = f"{sample}_" + df["to_cell"].astype(str)

        # Each interaction contributes a count of 1 at the sending spot and
        # a count of 1 at the receiving spot for that LR pair.
        from_counts = (
            df.groupby(["from_cell", "lr_pair"]).size().rename("count").reset_index()
            .rename(columns={"from_cell": "barcode"})
        )
        to_counts = (
            df.groupby(["to_cell", "lr_pair"]).size().rename("count").reset_index()
            .rename(columns={"to_cell": "barcode"})
        )
        long_records.append(pd.concat([from_counts, to_counts], ignore_index=True))

        for bc in pd.concat([df["from_cell"], df["to_cell"]]).unique():
            bc_to_sample.setdefault(bc, sample)

        print(
            f"{sample}: {df['from_cell'].nunique():,} sending, "
            f"{df['to_cell'].nunique():,} receiving spots; "
            f"{df['lr_pair'].nunique():,} unique LR pairs"
        )

    if not long_records:
        raise SystemExit(
            "No CellNEST files were successfully read. Check --nest-dir, "
            "--samples, and --file-pattern."
        )

    long_df = pd.concat(long_records, ignore_index=True)
    long_df = long_df.groupby(["barcode", "lr_pair"], as_index=False)["count"].sum()
    lr_matrix = (
        long_df.pivot(index="barcode", columns="lr_pair", values="count")
        .fillna(0)
        .astype(int)
    )
    lr_matrix.index.name = "barcode"
    lr_matrix.columns.name = None

    print(
        f"\nRaw matrix: {lr_matrix.shape[0]:,} spots × "
        f"{lr_matrix.shape[1]:,} LR pairs"
    )
    return lr_matrix, bc_to_sample


# ─────────────────────────────────────────────────────────────────────────────
# Join AnnData metadata onto the matrix index
# ─────────────────────────────────────────────────────────────────────────────

def load_obs(adata_path: Path, barcode_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adata.obs and return two lookup frames, one indexed by the
    prefixed barcode and one indexed by the plain barcode (i.e. with the
    '<exp>_<slide>_' prefix stripped).

    Falling back to the plain barcode covers the case where the AnnData
    barcodes don't carry the sample prefix.
    """
    adata = ad.read_h5ad(adata_path)
    obs = adata.obs.copy()

    if barcode_col in obs.columns:
        # The column already exists; reset_index() would collide on the
        # index name. Drop the index instead and keep the column.
        obs = obs.reset_index(drop=True)
    else:
        # Pull the index in as a column. If the index has no name, give it
        # the barcode column name we expect downstream.
        if obs.index.name is None:
            obs.index.name = barcode_col
        obs = obs.reset_index()
        if barcode_col not in obs.columns:
            # The index had a different name; alias the first column.
            first_col = obs.columns[0]
            print(
                f"Column '{barcode_col}' not on adata.obs; "
                f"falling back to the .obs index ('{first_col}')."
            )
            obs = obs.rename(columns={first_col: barcode_col})

    obs_by_prefixed = obs.set_index(barcode_col)
    obs_by_plain = obs.copy()
    obs_by_plain.index = obs_by_plain[barcode_col].str.replace(
        r"^[^_]+_[^_]+_", "", regex=True
    )
    return obs_by_prefixed, obs_by_plain


def attach_metadata(
    lr_matrix: pd.DataFrame,
    bc_to_sample: dict[str, str],
    adata_path: Path,
    barcode_col: str,
    keep_cols: list[str],
) -> pd.DataFrame:
    """Build the per-spot metadata frame keyed by the matrix's barcode index."""
    obs_by_prefixed, obs_by_plain = load_obs(adata_path, barcode_col)

    spot_meta = pd.DataFrame({"sample": bc_to_sample}, index=lr_matrix.index)
    spot_meta.index.name = "barcode"

    available = [c for c in keep_cols if c in obs_by_prefixed.columns]
    missing = [c for c in keep_cols if c not in obs_by_prefixed.columns]
    if missing:
        print(f"Missing in adata.obs (skipping): {missing}")

    for col in available:
        mapped = lr_matrix.index.map(obs_by_prefixed[col].to_dict())
        if pd.isna(mapped).mean() > 0.5:
            # Most barcodes didn't match — try the plain-barcode lookup.
            mapped = lr_matrix.index.map(obs_by_plain[col].to_dict())
        spot_meta[col] = mapped.values

    return spot_meta


def assign_tumour_stage(
    spot_meta: pd.DataFrame,
    coarse_col: str,
    prob_col: str,
    thresholds: tuple[float, float],
    out_col: str = "tumour_stage",
) -> pd.DataFrame:
    """Add a tumour_stage column: Early / Intermediate / Late / Non-tumour."""
    if coarse_col not in spot_meta.columns or prob_col not in spot_meta.columns:
        print(
            f"Cannot assign tumour stages — missing "
            f"'{coarse_col}' or '{prob_col}' from metadata."
        )
        spot_meta[out_col] = np.nan
        return spot_meta

    early_cut, late_cut = thresholds

    def _stage(row: pd.Series) -> str | float:
        if row.get(coarse_col) != "Tumour":
            return "Non-tumour"
        p = row.get(prob_col)
        if pd.isna(p):
            return np.nan
        if p <= early_cut:
            return "Early"
        if p <= late_cut:
            return "Intermediate"
        return "Late"

    spot_meta[out_col] = spot_meta.apply(_stage, axis=1)
    return spot_meta


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition outputs
# ─────────────────────────────────────────────────────────────────────────────

def write_condition(
    lr_matrix_raw: pd.DataFrame,
    spot_meta_base: pd.DataFrame,
    label: str,
    min_count: int,
    outdir: Path,
) -> None:
    """Apply the column-sum filter, recompute per-spot summary stats on the
    filtered matrix, and write both outputs to <outdir>/<label>/.
    """
    if min_count > 0:
        lr_matrix = lr_matrix_raw.loc[:, lr_matrix_raw.sum(axis=0) >= min_count]
    else:
        lr_matrix = lr_matrix_raw.copy()

    print(
        f"\n[{label}] LR pairs: {lr_matrix_raw.shape[1]:,} → "
        f"{lr_matrix.shape[1]:,}  (min_count = {min_count})"
    )
    print(f"[{label}] Spots: {lr_matrix.shape[0]:,}")

    # Recompute the per-spot summary stats on the filtered matrix; these may
    # legitimately differ from the unfiltered values.
    spot_meta_cond = spot_meta_base.drop(
        columns=["total_lr_count", "lr_pair_richness", "hill_q1", "normalized_hill"],
        errors="ignore",
    ).copy()
    spot_meta_cond["total_lr_count"] = lr_matrix.sum(axis=1).values
    spot_meta_cond["lr_pair_richness"] = (lr_matrix > 0).sum(axis=1).values
    spot_meta_cond["hill_q1"] = hill_q1_per_spot(lr_matrix).values
    spot_meta_cond["normalized_hill"] = (
        spot_meta_cond["hill_q1"]
        / spot_meta_cond["lr_pair_richness"].replace(0, np.nan)
    ).fillna(0)

    cond_dir = outdir / label
    cond_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cond_dir / f"lr_count_matrix_{label}.csv.gz"
    meta_path = cond_dir / f"spot_meta_{label}.csv"

    lr_matrix.to_csv(matrix_path)
    spot_meta_cond.to_csv(meta_path)
    print(f"[{label}] Wrote {matrix_path}")
    print(f"[{label}] Wrote {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI and main()
# ─────────────────────────────────────────────────────────────────────────────

def parse_condition(token: str) -> tuple[str, int]:
    """Parse a 'name=count' token from the --conditions flag."""
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            f"--conditions entries must be 'name=min_count', got '{token}'"
        )
    name, count = token.split("=", 1)
    try:
        return name, int(count)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"min_count for '{name}' must be an integer, got '{count}'"
        ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--nest-dir", type=Path, required=True,
                   help="Directory containing CellNEST per-sample CSVs.")
    p.add_argument("--adata", type=Path, required=True,
                   help="Path to the AnnData (.h5ad) holding spot metadata.")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Top-level output directory. Per-condition subdirs "
                        "are created inside.")
    p.add_argument("--samples", nargs="+", required=True,
                   help="Sample IDs to process (e.g. exp1_B1 exp1_C1 ...).")
    p.add_argument("--file-pattern",
                   default="NEST_PDAC_{sample}_manualDB_thresholded.csv",
                   help="CellNEST filename pattern with '{sample}' placeholder.")
    p.add_argument("--conditions", nargs="+", type=parse_condition,
                   default=[("unfiltered", 0), ("filtered", 50)],
                   help="Filter conditions as 'label=min_count' tokens. "
                        "Default: unfiltered=0 filtered=50.")
    # ── adata.obs column names (defaults match the upstream notebook) ────────
    p.add_argument("--barcode-col", default="prefixed_barcode")
    p.add_argument("--tradeseq-col", default="tradeseq_cluster")
    p.add_argument("--nongland-prob-col", default="Non-glandular probability")
    p.add_argument("--morphology-col", default="full")
    p.add_argument("--coarse-col", default="coarse")
    # ── Tumour stage thresholds ──────────────────────────────────────────────
    p.add_argument("--stage-thresholds", nargs=2, type=float,
                   default=[0.178861, 0.495522],
                   metavar=("EARLY_CUT", "LATE_CUT"),
                   help="Non-glandular-probability cut-points separating "
                        "Early / Intermediate / Late tumour stages "
                        "(default: 0.178861 0.495522).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Building spot x LR matrix from CellNEST outputs:")
    lr_matrix_raw, bc_to_sample = build_lr_matrix(
        args.samples, args.nest_dir, args.file_pattern
    )

    print("\nAttaching per-spot metadata from AnnData:")
    keep_cols = [
        args.nongland_prob_col,
        args.tradeseq_col,
        args.morphology_col,
        args.coarse_col,
        "n_umis",
        "n_genes",
    ]
    spot_meta_base = attach_metadata(
        lr_matrix_raw,
        bc_to_sample,
        args.adata,
        barcode_col=args.barcode_col,
        keep_cols=keep_cols,
    )

    print("\nAssigning tumour stages:")
    spot_meta_base = assign_tumour_stage(
        spot_meta_base,
        coarse_col=args.coarse_col,
        prob_col=args.nongland_prob_col,
        thresholds=tuple(args.stage_thresholds),
    )

    stage_counts = spot_meta_base.get("tumour_stage", pd.Series(dtype=object)).value_counts()
    if not stage_counts.empty:
        print(f"  tumour stage counts:\n{stage_counts.to_string()}")
    for col in [args.tradeseq_col, args.nongland_prob_col, "tumour_stage"]:
        if col in spot_meta_base.columns:
            pct = spot_meta_base[col].notna().mean() * 100
            print(f"  {col} coverage: {pct:.1f}%")

    # Save a "raw" copy of the unfiltered combined outputs at the top level
    # for downstream scripts that don't want per-condition slicing.
    raw_matrix_path = args.outdir / "lr_count_matrix_raw.csv.gz"
    raw_meta_path = args.outdir / "spot_meta.csv"
    lr_matrix_raw.to_csv(raw_matrix_path)
    spot_meta_base.to_csv(raw_meta_path)
    print(f"\n Wrote raw matrix → {raw_matrix_path}")
    print(f"Wrote raw meta   → {raw_meta_path}")

    print("\nWriting per-condition outputs:")
    for label, min_count in args.conditions:
        write_condition(
            lr_matrix_raw=lr_matrix_raw,
            spot_meta_base=spot_meta_base,
            label=label,
            min_count=min_count,
            outdir=args.outdir,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())