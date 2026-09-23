# Individual Differences in Object Representations

Modeling and analysis code for the manuscript on individual differences in
object representations. The pipeline covers Study 1 (weighted-embedding models
and their analysis) and Study 2 (a triplet study run in jATOS, plus analysis of
the combined data).

Modeling and analysis are in Python and R. The experiment is built with jsPsych
and custom JavaScript/HTML/CSS.

> **Naming note:** the folder `data/study1-2025-08` holds the data we collected
> ourselves; this data corresponds to "Study 2" in the manuscript.

## Setup

Follow these in order. Steps 1–2 are prerequisites you install once; steps 3–4
are run from the repo root and set up the environment and data.

**1. Install uv** — the Python environment/package manager used throughout.
See [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) for the installer command and
Windows PATH notes. Nothing below works without it.

**2. Install 7-Zip** (https://www.7-zip.org) before fetching the data. The image
archive has ~28,000 files; with 7-Zip, unpacking takes about 15 minutes on a
normal laptop. Without it the download script falls back to Python's built-in
unzip, whose encrypted-zip path is far slower (potentially hours).

**3. Create the Python environment** (from the repo root):

```bash
uv sync --extra cpu    # or --extra cu128 / cu130 for an NVIDIA GPU
```

This installs the pinned Python 3.12 environment with all packages. Everything
below is run through `uv run`, which uses that environment without a separate
activation step. Details in [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md).

**4. Download the data** (from the repo root):

```bash
uv run python get_data.py
```

This downloads and verifies all external data from OSF into `data/` in one step
(four source files plus the ~5 GB image archive, which is password-protected and
unpacked automatically). Nothing is downloaded or unzipped by hand. Details in
[`DATA_SETUP.md`](DATA_SETUP.md).

### Additional prerequisites (only for the parts that need them)

- **R** with **RStudio**, plus the **`rutils`** package from the main author
  (github.com/MirkoTh/rutils; install via devtools/Rtools) for the R scripts.
  Open the `.Rproj` file in the repo root in RStudio before running any R
  script — this sets the working directory to the repo root so the scripts'
  relative paths resolve. Run each R script from within RStudio (open it and
  Source / Run).
- **jATOS** (www.jatos.org), installed locally, to run the experiment. If you
  only want to run Study 2 without the modeling, a ready `.jzip` is available at
  https://osf.io/m4yfr/overview (import it directly if you have a mindprobe
  account).

> Running the models can take a long time. The repo also provides the resulting
> files from the modeling scripts, so the analyses can be run without retraining.

## Study 1

Prepare the model inputs, then train the weighted-embedding (PyTorch) models and
analyze them.

1. **Prepare triplets:** run `R/things-triplets.R` in RStudio — also writes the
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
> earlier in the project and will not be re-created identically on a rerun —
> reasons can include different seeds, model architecture or dimensionality, or a
> different training dataset. Batch-shuffling has since been made seed-deterministic
> for future runs, but this does not reconstruct the original set; the published
> Study 2 is unchanged. Overlap for the model-based half (220 triplets) is
> substantial: 166/220, 171/220, 163/220 for dimensionalities 25, 30, 35.

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

1. **Load:** in RStudio, run in order —
   `exclusion-criteria.R`, `concatenate-ooo-old-new.R`, `R/EDA.R`.
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

1. run `R/plot-figures-ms.R` in RStudio
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
