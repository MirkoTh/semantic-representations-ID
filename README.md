# Individual Differences in Object Representations

Modeling and analysis code for the manuscript on individual differences in
object representations. The pipeline covers Study 1 (weighted-embedding models
and their analysis) and Study 2 (a triplet study run in jATOS, plus analysis of
the combined data).

Modeling and analysis are in Python and R. The experiment is built with jsPsych
and custom JavaScript/HTML/CSS.

> **Naming note:** "Study 2" in the manuscript corresponds to `study1` in the
> folder structure. This mismatch runs throughout the repo — keep it in mind
> whenever a path says `study1`.

## Setup

Two one-time steps. Both are covered in detail in the linked files.

**1. Python environment** — see [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md).

```bash
uv sync --extra cpu    # or --extra cu128 / cu130 for an NVIDIA GPU
```

This installs the pinned Python 3.12 environment with all packages. Everything
below is run through `uv run`, which uses that environment without a separate
activation step.

**2. Data** — see [`DATA_SETUP.md`](DATA_SETUP.md).

```bash
uv run python get_data.py
```

This downloads and verifies all external data from OSF into `data/` in one step
(four source files plus the ~5 GB image archive, which is password-protected and
unpacked automatically). Nothing is downloaded or unzipped by hand.

> **Install 7-Zip first** (https://www.7-zip.org). The image archive has ~28,000
> files; with 7-Zip, unpacking takes about 15 minutes on a normal laptop.
> Without 7-Zip the script falls back to Python's built-in unzip, whose
> encrypted-zip path is far slower (potentially hours).

### Other prerequisites

- **jATOS** (www.jatos.org), installed locally, to run the experiment. If you
  only want to run Study 2 without the modeling, a ready `.jzip` is available at
  https://osf.io/m4yfr/overview (import it directly if you have a mindprobe
  account).
- **R** and the **`rutils`** package from the main author
  (github.com/MirkoTh/rutils; install via devtools/Rtools) for the R scripts.

> Running the models can take a long time. The repo also provides the resulting
> files from the modeling scripts, so the analyses can be run without retraining.

## Study 1

Prepare the model inputs, then train the weighted-embedding (PyTorch) models and
analyze them.

1. **Prepare triplets:** `uv run Rscript R/things-triplets.R` — also writes the
   "diagnostic triplets" (those observed multiple times), needed later for the
   Study 2 triplet set.
2. **Train models:**
   - Hyperparameter search on lambda: `uv run python initialize-model-highdim.py`
     (can be split across machines by lambda value to run faster)
   - Individual-differences effects over dimensionality:
     `uv run python initialize-model-improvement-dimensionality.py`
   - Split-half reliabilities:
     `uv run python initialize-model-splithalf-reliability-icc.py` and
     `uv run python initialize-model-splithalf-reliability.py`
3. **Analyze** (Jupyter notebooks, launch with `uv run jupyter lab`):
   - `analyze-highdim-model.ipynb` — the hyperparameter search
   - `dimensionality-accuracy-improvement.ipynb` — how dimensionality affects
     the individual-differences improvement
   - `split-half-reliability.ipynb` — the split-half reliabilities

## Study 2

First build the fixed triplet set, then run the study in jATOS, then analyze the
combined data.

### Build the triplet set

The exact triplet IDs used in the published Study 2 are saved in
`data/triplets-delta_USED_STUDY2.csv`. To reproduce our results with the same
440 triplets, rename that file to `data/triplets-delta.csv` and **skip the
"Model deltas" step** below.

> **On reproducing the triplet set from scratch:** the published set was saved
> earlier in the project and will not be re-created identically if you rerun the
> pipeline, because the batch-shuffling in earlier runs drew from the global RNG
> and so drifted with changes to seed, model architecture, and dimensionality.
> The shuffling is now seed-deterministic going forward (see the modeling code),
> but this does not retroactively reconstruct the original set — the published
> Study 2 was already run and is not changed. For the model-based half (220 of
> the 440 triplets), the overlap with the original is substantial: 166/220,
> 171/220, and 163/220 for dimensionalities 25, 30, and 35 respectively.

Steps:

1. **Model deltas** (skip if reusing the original triplets):
   `analyze-model-deltas.ipynb` — computes prediction-accuracy differences
   between average and idiosyncratically weighted representations; saves
   `data/triplets-delta.csv`.
2. **Create triplet set:** `create-triplet-set.ipynb` — loads
   `data/triplets-delta.csv` and builds the 440-triplet set (220 model-based,
   220 random), saving the files needed to run the study in jATOS.

### Run the study

The jATOS study code lives in a separate folder. To rebuild it from the analysis
results, copy `experiments/` and `data/` (after running the code above) into the
jATOS `study_assets_root`, then run `delete-unused-images.ipynb` there. See
www.jatos.org for jATOS details.

### Load and analyze the data

1. **Load:** run in order —
   `uv run Rscript exclusion-criteria.R`,
   `uv run Rscript concatenate-ooo-old-new.R`,
   `uv run Rscript R/EDA.R`.
   These apply exclusion criteria, concatenate the new results with the source
   study, and save per-participant average triplet response times. Raw Prolific
   files are not provided — only data with hashed Prolific IDs.
2. **Analyze** (notebooks via `uv run jupyter lab`):
   - `analyze-combined-data-finaldym.ipynb` — dimensionality 35
   - `analyze-combined-data-model-alldims.ipynb` — all 12 dimensionalities
   - `predict-dims-by-interests.ipynb` — predicting dimensional weightings from
     self-reported work history and interests

## Figures

After running the models and analyses above, plot all manuscript figures with:

1. `uv run Rscript R/plot-figures-ms.R`
2. `plot-figures-ms.ipynb` (via `uv run jupyter lab`)

The study-overview figures and the first result figure were assembled manually
from these outputs.

## Repository layout

Data setup and integrity: `get_data.py`, `DATA_SETUP.md`. Environment:
`pyproject.toml`, `uv.lock`, `.python-version`, `ENVIRONMENT_SETUP.md`. Modeling:
`initialize-model-*.py` (run configs) drive `run-embedding-decision-combined-data.py`
using the models in `models/model.py` and helpers in `utils.py`; plotting
helpers in `plotting.py`. Analysis notebooks and R scripts sit at the repo root
and under `R/`.
