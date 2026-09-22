# Environment setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment, dependencies, and Python version itself. You do **not** need conda,
pyenv, or a manual virtualenv — uv handles all of it, and it never installs into
your system Python.

Also please note the information on troubleshooting at the end of this file.

## 1. Install uv (once, system-wide)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv is a single self-contained binary with no Python dependency of its own.

## 2. Create the environment

Run all `uv` commands from the **repository root** — the folder that contains
`pyproject.toml`. (uv searches upward from your current directory for a
`pyproject.toml`, so a subfolder of the repo also works, but anywhere outside
the repo will not find the project.)

```bash
cd /path/to/this-repo   # the directory containing pyproject.toml
```

Pick **one** compute backend for PyTorch. If you're unsure, use `cpu` — the
models produce identical results on CPU, just slower.

```bash
# DEFAULT — works on any machine, no GPU required:
uv sync --extra cpu

# NVIDIA GPU with CUDA 12.8 drivers:
uv sync --extra cu128

# NVIDIA GPU with CUDA 13.0 drivers (current PyTorch PyPI stable):
uv sync --extra cu130
```

This creates a `.venv/` in the project root and installs everything from the
lockfile. The correct Python version (3.12, pinned in `.python-version`) is
downloaded automatically by uv if you don't already have it — no pyenv needed.

The three backends are mutually exclusive; `uv sync --extra cu128` replaces a
prior `cpu` install in the same environment (and vice versa).

## 3. Run things

No `activate` step is needed — prefix commands with `uv run`:

```bash
uv run python get_data.py                 # fetch the data (see DATA_SETUP.md)
uv run python initialize-model-highdim.py # run a model
uv run jupyter lab                        # launch notebooks in the env
```

If you prefer a classic activated shell, `source .venv/bin/activate` still works.

## How reproducibility is guaranteed

- **`uv.lock`** pins every dependency, including all transitive ones, with
  hashes. Commit it. A collaborator who runs `uv sync --extra cpu` gets a
  byte-for-byte identical set of packages.
- **`.python-version`** pins the interpreter (3.12), so everyone runs the same
  Python, not "whatever `python` happens to be on the machine."
- **`pyproject.toml`** declares the high-level dependencies and the PyTorch
  index routing. This is the file you edit (via `uv add <pkg>`); the lockfile is
  generated.

Together with `data/osf.lock.json` (data integrity) this pins all three moving
parts: the interpreter, the packages, and the input data.

## Adding or changing dependencies

```bash
uv add <package>            # add a runtime dependency
uv add --dev <package>      # add a dev-only dependency
uv remove <package>         # drop one
uv lock --upgrade           # re-resolve to newest allowed versions
```

Each of these updates `pyproject.toml` and `uv.lock`; commit both.

## Notes on the PyTorch setup

PyTorch CUDA wheels come from PyTorch's own package index, not PyPI, so
`pyproject.toml` declares three explicit indexes (`pytorch-cpu`,
`pytorch-cu128`, `pytorch-cu130`) and routes `torch` to the right one based on
which extra you selected. On macOS there are no CUDA wheels, so the `cuNNN`
extras fall back to the default index there automatically. If you need a
different CUDA version, add another index block and extra following the same
pattern (valid targets are listed at
https://download.pytorch.org/whl/ — e.g. `cu126`).

## The R side

Parts of this project (data prep, EDA, figures) run in R, which uv does not
manage. Install R separately and the `rutils` package from
github.com/MirkoTh/rutils as described in the main code instructions. If you
want the R dependencies pinned the same way uv pins Python, consider adding
`renv` to the R project — say the word and that can be set up too.

## Troubleshooting: "uv is not recognized" (Windows)

If `uv --version` fails in a new terminal right after installing, it's almost
always PATH, not a broken install. The installer adds uv to your PATH, but the
change only takes effect in terminals opened **afterwards** — so close all
terminals (including IDE-integrated ones, which keep the PATH from when the IDE
started) and open a fresh PowerShell from the Start menu. If it still fails,
restart the IDE entirely.

For anything beyond that, see the official docs and a Windows-specific walkthrough:

- https://docs.astral.sh/uv/getting-started/installation/
- https://pydevtools.com/handbook/how-to/how-to-install-uv-on-windows/

Quick check that uv is installed and only PATH is the issue:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" --version   # prints a version -> uv is fine
```
