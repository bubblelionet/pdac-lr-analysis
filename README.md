# Ordinal regression of tradeSeq trajectory cluster from LR profiles

This pipeline includes two main scripts:

1. **`build_spot_lr_inputs.py`** — builds the spot × LR count matrix and per-spot metadata from raw CellNEST output CSVs and an AnnData file.
2. **`run_ordinal_regression.py`** — fits the ordinal regression on the outputs of step 1.

The two scripts together reproduce the trajectory-classification analysis: from `NEST_PDAC_<sample>_manualDB_thresholded.csv` files all the way to coefficient tables, threshold tables, and cross-validation accuracy.

## Quick start example:

```bash
# Step 1 — build inputs
python build_spot_lr_inputs.py \
    --nest-dir  /path/to/CellNEST_outputs \
    --adata     /path/to/institutional_samples_untreated.h5ad \
    --outdir    /path/to/spot_level_analysis \
    --samples   exp1_B1 exp1_C1 exp1_D1 \
                exp2_A1 exp2_B1 exp2_D1 \
                exp4_A1 exp4_C1 exp4_D1 \
                exp6_A1 exp6_B1 exp6_D1

# Step 2 — fit ordinal regression on the filtered condition
python run_ordinal_regression.py \
    --lr-matrix /path/to/spot_level_analysis/filtered/lr_count_matrix_filtered.csv.gz \
    --spot-meta /path/to/spot_level_analysis/filtered/spot_meta_filtered.csv \
    --outdir    /path/to/spot_level_analysis/filtered/ordinal \
    --prefix    ordinal_filtered_tradeseq
```

---

## `build_spot_lr_inputs.py`

Builds the two inputs that `run_ordinal_regression.py` consumes, starting from raw CellNEST outputs.

### What it does

1. For each sample, reads `NEST_PDAC_<sample>_manualDB_thresholded.csv` from `--nest-dir`.
2. Builds an LR pair name as `<ligand>-<receptor>`, prefixes each spot barcode with the sample ID, and counts the number of interactions in which each (spot, LR pair) appears as either sender or receiver. *Each interaction therefore contributes a count of 1 to the sending spot and a count of 1 to the receiving spot for that LR pair* — matching the upstream notebook's semantics.
3. Pivots the long-format counts into a spot × LR-pair matrix.
4. Loads the adata and joins per-spot metadata (trajectory label, morphology, coarse annotation, etc.) onto the matrix's barcode index. Falls back to a plain-barcode lookup if the adata barcodes don't carry the sample prefix.
5. Computes per-spot summary statistics: total LR count, LR-pair richness, Hill number of order 1, and normalized Hill.
6. Assigns each spot a tumour stage (Early / Intermediate / Late) based on the coarse pathologist annotation and the predicted non-glandular probability. Spots not annotated as Tumour get `Non-tumour`; spots with a missing probability get `NaN`.
7. Writes a "raw" pair of outputs at the top of `--outdir`, then writes one filtered pair per condition under `--outdir/<condition>/`.

### Required CellNEST CSV columns

The script needs at least: `from_cell`, `to_cell`, `ligand`, `receptor`. Other columns (`edge_rank`, `component`, `attention_score`, etc.) are ignored.

### Required adata columns

Defaults match the upstream notebook config:

| Default name | What it holds |
|---|---|
| `prefixed_barcode` | Sample-prefixed spot barcode (`<exp>_<slide>_<barcode>`). |
| `sample` | Sample ID. |
| `tradeseq_cluster` | Trajectory label: `Low`, `Intermediate`, `High`, or `Other`. |
| `Non-glandular probability` | Continuous predicted probability of the non-glandular morphology class. |
| `coarse` | Coarse pathologist label; only `Tumour` spots get a tumour stage. |
| `full` | Full pathologist label (used downstream for plotting). |

Each of these can be overridden via CLI flags (see below). Missing columns are skipped with a warning rather than crashing.

### Output files

```
<outdir>/
├── lr_count_matrix_raw.csv.gz          # all spots, all LR pairs, unfiltered
├── spot_meta.csv                        # all spots, all metadata, unfiltered
├── unfiltered/
│   ├── lr_count_matrix_unfiltered.csv.gz
│   └── spot_meta_unfiltered.csv
└── filtered/
    ├── lr_count_matrix_filtered.csv.gz
    └── spot_meta_filtered.csv
```

The per-condition matrix differs from the raw one in two ways: (1) LR pairs with `column sum < min_count` are dropped, and (2) per-spot summary statistics (`total_lr_count`, `hill_q1`, etc.) are recomputed on the filtered matrix, so they're consistent with the columns that remain.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--nest-dir` | required | Directory containing per-sample CellNEST CSVs. |
| `--adata` | required | Path to the AnnData (.h5ad) with spot metadata. |
| `--outdir` | required | Top-level output directory (created if absent). |
| `--samples` | required | Space-separated list of sample IDs (e.g. `exp1_B1 exp1_C1 …`). |
| `--file-pattern` | `NEST_PDAC_{sample}_manualDB_thresholded.csv` | Filename template; `{sample}` is substituted. |
| `--conditions` | `unfiltered=0 filtered=50` | Space-separated `label=min_count` tokens. Each becomes one output subdir. |
| `--barcode-col` | `prefixed_barcode` | AnnData column holding the prefixed barcode. |
| `--tradeseq-col` | `tradeseq_cluster` | AnnData column with trajectory labels. |
| `--nongland-prob-col` | `Non-glandular probability` | AnnData column for stage thresholding. |
| `--morphology-col` | `full` | Full-resolution morphology label. |
| `--coarse-col` | `coarse` | Coarse-resolution morphology label. |
| `--stage-thresholds` | `0.178861 0.495522` | Two cut-points on non-glandular probability separating Early/Intermediate/Late. |

### Notes

- Counts are **per-endpoint, not per-interaction**: an interaction with `from_cell=A` and `to_cell=B` adds 1 to A's count and 1 to B's count for that LR pair. An autocrine interaction (`A → A`) therefore adds 2 to A.
- The barcode-prefixing avoids collisions when the same raw barcode appears in two different samples.
- If you change the trajectory ordering, edit `STAGE_ORDER` in `run_ordinal_regression.py` (not here — this script doesn't need to know the ordering).

---

## `run_ordinal_regression.py`

Predicts the ordered tradeSeq trajectory cluster (`Low < Intermediate < High`) from per-spot ligand–receptor (LR) interaction counts using a proportional-odds (cumulative-logit) ordinal regression.

### What it does

1. Loads a per-spot LR count matrix and a per-spot metadata table (the outputs of `build_spot_lr_inputs.py`, both indexed by barcode).
2. Restricts to spots with a trajectory label in `{Low, Intermediate, High}` (drops `Other` and `NaN`).
3. Drops zero-variance LR pairs in that subset.
4. Standardizes the LR features (zero mean, unit variance) and integer-encodes the labels.
5. Fits a `statsmodels.OrderedModel` with a logit link via BFGS, jointly estimating one coefficient per LR pair and two ordered thresholds.
6. Performs 5-fold stratified cross-validation, refitting the scaler and the model within each training fold to avoid leakage. Each test spot is assigned the class with the highest posterior probability.
7. Saves:
   - `*_coefs.csv` — coefficient, p-value, and direction for every LR pair, sorted by `|coef|`.
   - `*_thresholds.csv` — the two latent-scale cut-points with their standard errors.
   - `*_cv_accuracies.csv` — per-fold balanced accuracy.
   - `*_coefs.html`, `*_thresholds.html`, `*_cv.html` — Altair plots (skip with `--no-plots`).


### All flags

| Flag | Default | Description |
|---|---|---|
| `--lr-matrix` | required | Path to the spot × LR-pair count CSV (may be gzipped). |
| `--spot-meta` | required | Path to per-spot metadata CSV indexed by barcode. |
| `--outdir` | required | Output directory; created if absent. |
| `--tradeseq-col` | `tradeseq_cluster` | Column in metadata holding the trajectory labels. |
| `--prefix` | `ordinal_tradeseq` | Prefix for all output filenames. |
| `--top-n` | `20` | Number of LR pairs in the coefficient bar chart. |
| `--seed` | `42` | Random seed for the 5-fold CV split. |
| `--min-spots` | `30` | Minimum labelled spots needed to fit. |
| `--no-plots` | off | Skip HTML plot generation. |

### Output interpretation

**`*_coefs.csv`** — one row per LR pair:

| Column | Meaning |
|---|---|
| `lr_pair` | Ligand–receptor pair name. |
| `coef` | Standardized coefficient on the latent scale. Positive → higher abundance shifts a spot toward later trajectory states (High). Negative → toward Low. |
| `pval` | Wald p-value from the fit. |
| `abs_coef` | Absolute coefficient, used for ranking. |
| `sig` | `True` if `pval < 0.05`. |
| `direction` | Human-readable direction label. |

Rows are sorted by `abs_coef` descending.

**`*_thresholds.csv`** — two rows, one per class boundary, giving the cut-point on the latent scale together with its standard error. The two cut-points partition the latent line into the three predicted classes.

**`*_cv_accuracies.csv`** — per-fold balanced accuracy from 5-fold stratified CV. Chance for a 3-class problem is `1/3 ≈ 0.333`.


---

## Installation

Python ≥ 3.9. Install dependencies into your environment:

```bash
pip install -r requirements.txt
```

## Reproducing the original analysis

Victoria ran the regression for two filter conditions (`unfiltered`, `filtered@50`) and three subsets (`all`, `tumour`, `nontumour`); only the `tumour` and `all` subsets had a non-trivial mix of trajectory labels, so the regression is informative there. `run_ordinal_regression.py` handles one (matrix, meta) pair per run — invoke it once per condition × subset combination you want to evaluate.

1. Loads a per-spot LR count matrix and a per-spot metadata table (both indexed by barcode).
2. Restricts to spots with a trajectory label in `{Low, Intermediate, High}` (drops `Other` and `NaN`).
3. Drops zero-variance LR pairs in that subset.
4. Standardizes the LR features (zero mean, unit variance) and integer-encodes the labels.
5. Fits a `statsmodels.OrderedModel` with a logit link via BFGS, jointly estimating one coefficient per LR pair and two ordered thresholds.
6. Performs 5-fold stratified cross-validation, refitting the scaler and the model within each training fold to avoid leakage. Each test spot is assigned the class with the highest posterior probability.
7. Saves:
   - `*_coefs.csv` — coefficient, p-value, and direction for every LR pair, sorted by `|coef|`.
   - `*_thresholds.csv` — the two latent-scale cut-points with their standard errors.
   - `*_cv_accuracies.csv` — per-fold balanced accuracy.
   - `*_coefs.html`, `*_thresholds.html`, `*_cv.html` — Altair plots (skip with `--no-plots`).


## Inputs

The script consumes the per-condition outputs produced by the upstream spot-level pipeline:

- `lr_count_matrix_<condition>.csv.gz` — rows = spots (prefixed barcode), columns = LR pairs (e.g. `LIGAND-RECEPTOR`), values = integer counts.
- `spot_meta_<condition>.csv` — rows = spots, must include a column with the trajectory label (default: `tradeseq_cluster`).

Both files must use the same barcode index. The script intersects them automatically and reports if any barcodes are dropped.

## Installation

Python ≥ 3.9. Install dependencies into your environment:

```bash
pip install numpy pandas scikit-learn statsmodels altair
```

## Usage

```bash
python run_ordinal_regression.py \
    --lr-matrix /path/to/lr_count_matrix_filtered.csv.gz \
    --spot-meta /path/to/spot_meta_filtered.csv \
    --outdir    ./ordinal_filtered \
    --prefix    ordinal_filtered_tradeseq
```

For the `unfiltered` condition, swap in the unfiltered files and a different `--outdir` / `--prefix`.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--lr-matrix` | required | Path to the spot × LR-pair count CSV (may be gzipped). |
| `--spot-meta` | required | Path to per-spot metadata CSV indexed by barcode. |
| `--outdir` | required | Output directory; created if absent. |
| `--tradeseq-col` | `tradeseq_cluster` | Column in metadata holding the trajectory labels. |
| `--prefix` | `ordinal_tradeseq` | Prefix for all output filenames. |
| `--top-n` | `20` | Number of LR pairs in the coefficient bar chart. |
| `--seed` | `42` | Random seed for the 5-fold CV split. |
| `--min-spots` | `30` | Minimum labelled spots needed to fit. |
| `--no-plots` | off | Skip HTML plot generation. |

## What each output means

**`*_coefs.csv`** — one row per LR pair:

| Column | Meaning |
|---|---|
| `lr_pair` | Ligand–receptor pair name. |
| `coef` | Standardized coefficient on the latent scale. Positive → higher abundance shifts a spot toward later trajectory states (High). Negative → toward Low. |
| `pval` | Wald p-value from the fit. |
| `abs_coef` | Absolute coefficient, used for ranking. |
| `sig` | `True` if `pval < 0.05`. |
| `direction` | Human-readable direction label. |

Rows are sorted by `abs_coef` descending.

**`*_thresholds.csv`** — two rows, one per class boundary, giving the cut-point on the latent scale together with its standard error. The two cut-points partition the latent line into the three predicted classes.

**`*_cv_accuracies.csv`** — per-fold balanced accuracy from 5-fold stratified CV. Chance for a 3-class problem is `1/3 ≈ 0.333`.

