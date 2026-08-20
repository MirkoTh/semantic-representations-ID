# Data setup

All external data for this project lives on OSF and is fetched by a single
script, `get_data.py`. You do not download or unzip anything by hand.

## Quick start

```bash
uv run python get_data.py
```

(Or, in an already-activated environment, just `python get_data.py`. The only
dependencies are `requests` and `tqdm`, both declared in `pyproject.toml`.)

This downloads into `data/`:

| File / folder                                         | OSF     | Source project |
| ----------------------------------------------------- | ------- | -------------- |
| `labels.mat`                                          | `h4m8a` | osf.io/z2784   |
| `words.mat`                                           | `qe8sj` | osf.io/z2784   |
| `unique_id.txt`                                       | `2y463` | osf.io/z2784   |
| `triplets_large_final_correctednc_correctedorder.csv` | `h2smy` | osf.io/f5rn6   |
| `images/` (~1854 category folders, ~28k images)       | `rdxy2` | osf.io/jum2f   |

Files that already exist in `data/` are skipped, so the script is safe to
re-run.

## The images archive

The images come from OSF as one large (~5 GB) **password-protected** zip and are
unpacked into `data/images/`.

- **Password:** `things4all` (published with the source dataset under https://osf.io/jum2f; hardcoded in
  the get_data.py — by running the script, you agree to use the images only for
  research and non-commercial purposes and to not distribute or alter them without
  permission of the owner of the images.).
- **7-Zip strongly recommended.** If `7z` is available (on PATH, or at the
  default Windows install path `C:\Program Files\7-Zip\7z.exe`), the script uses
  it — extraction takes a few minutes. Without 7-Zip it falls back to Python's
  built-in unzip, whose encrypted-zip path is pure-Python and can take _hours_.
  Install 7-Zip from https://www.7-zip.org before running if you don't have it.
- The archive nests everything under a top-level `object_images/` folder; the
  script flattens this so images land directly at `data/images/<category>/...`.

### Large, resumable download

The ~5 GB download **resumes automatically** if the connection drops: it streams
into a `.part` file and, on the next attempt, continues from where it left off
via an HTTP range request (up to 10 attempts). The final file is checked against
the server's reported size before being accepted, so a truncated download is
never treated as complete. If a run dies entirely, just run the script again —
it picks up the partial `data/images.zip.part` rather than restarting.

## Integrity verification

Every item is pinned to a known SHA-256 in the `HASHES` dict at the top of
`get_data.py`. After each download the script recomputes the hash and aborts on
any mismatch, so you always get exactly the bytes the analysis was built on —
this catches both download corruption and any future change to a file on OSF.

The `images/` entry is a **combined hash of the unpacked image files** (sorted
relative paths + file bytes folded into one digest), not a hash of the zip. The
zip wrapper's bytes aren't guaranteed stable across OSF downloads, whereas the
image files themselves are — so this verifies what actually matters and is
immune to zip-regeneration false positives. A missing, extra, renamed, or
altered image changes the combined hash.

## Re-pinning (only if upstream data changes)

The hashes are already filled in and committed. You only need this if an OSF
file is intentionally updated and you want to adopt the new version:

```bash
uv run python get_data.py --print-hashes
```

This downloads everything, prints a ready-to-paste `"name": "hash",` line per
item, and does not verify. Paste the new values into the `HASHES` dict, and
commit. This is the documented procedure for how the pins were produced.
