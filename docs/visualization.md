# Phase 1 Task 6: GPU trace visualization

Use Python 3.9–3.12. On this Mac, create an isolated environment with the
existing Apple Python; no Homebrew or new Python installation is needed:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-visualization.lock.txt
```

Run from the project root:

```bash
.venv/bin/python scripts/visualize_traces.py
```

Verify the generated gallery against its source arrays and hashes:

```bash
.venv/bin/python scripts/verify_visualizations.py
```

For a custom gallery, pass `--gallery /path/to/gallery` (and `--data-root` if
needed). This read-only command checks current gallery provenance; the recorded
verification report also includes the before/after dataset integrity audit.

The old `.venv/bin/python visualize_traces.py` command forwards to the same implementation.
The default data root is resolved from the script location, so neither command
requires a hard-coded home directory. No GPU, PyTorch, model, or network is needed.
Matplotlib uses the noninteractive Agg backend and a temporary cache directory.

Selection sorts official **training** records by path, chooses one distinct
trace with primary labeled bins for each observed L1 class, then four additional
traces with real multi-hot overlap. There is no random seed because no random
sampling is used. The default gallery has 15 plots. This is an illustrative
gallery, not a representative prevalence estimate. No validation/test NPZ files
are loaded. The CLI intentionally offers no validation/test split option.

```bash
# Inspect stored physically scaled values, without display normalization:
.venv/bin/python scripts/visualize_traces.py --normalization stored --output-dir outputs/phase1/stored

# Use another dataset root or output directory:
.venv/bin/python scripts/visualize_traces.py --data-root /path/to/dataset --output-dir /tmp/kernel-xray-plots --overlap-examples 4 --dpi 140
```

Each figure uses a shared time axis measured from the first timestamp edge:

1. All 24 counter channels, in original NPZ order (top to bottom).
2. Primary L1 ground truth, with gray for unlabeled bins. Colors match the
   named rows in the multi-hot panel.
3. All 12 L1 multi-hot rows. A colored cell marks that class as active;
   gray means inactive. Multiple colored rows in a column indicate overlap.
4. Number of active L1 classes. Red shading marks contiguous intervals with
   at least two active classes, also overlaid on the hardware-counter panel.

The default `p99` view divides each channel by its own absolute 99th percentile
and clips to [0, 1]. Zero-percentile channels use a divisor of 1. This improves
visibility but can hide spikes and prevents comparing absolute magnitudes
across channels or traces. `stored` displays the values as saved by the dataset;
these are already physically scaled upstream and are not raw hardware units.
No tensor is changed on disk. Overlap shading denotes label envelopes, not an
independent discovery from the counters or instruction-level timing evidence.

`manifest.json` records selected training paths, source input/label/split SHA-256
hashes, plot hashes, dependency versions, normalization divisors, labeled/overlap
bin counts, and exact overlap intervals in milliseconds. Selection and rendering
are deterministic for a fixed dataset and environment; font/library changes can
alter PNG bytes on another machine. Hashes identify the concrete generated files.
Use the pinned requirements for the Python 3.9-compatible environment. Older
gallery manifests retain their original environment versions. New outputs overwrite
matching gallery filenames; choose a new output directory when comparing runs.

The implementation checks tensor/label shapes, finite counters, increasing and
matching time edges, primary-mask consistency, binary multi-hot values,
primary-class inclusion in multi-hot labels, and agreement with overlap flags.
The verification report checks all generated manifest entries against the real
training data, confirms deterministic rerendering, and records unchanged hashes
for all 4,332 raw NPZ files and seven official split files. Split disjointness
checks read JSON path lists only. These checks do not rerun the earlier complete
semantic dataset validation; that original report is retained separately.

## Setup troubleshooting

`numpy==2.4.6` and the previous lock file targeted a newer Python environment.
The replacement pins NumPy 1.26.4 and Matplotlib 3.9.4, compatible with Python
3.9. The complete dependency set is pinned after installation verification.

If `python` is not found, use `.venv/bin/python` directly. If Matplotlib or NumPy
is missing, install using that same interpreter, not a different `pip` command.
The explicit environment path avoids accidentally using `/usr/bin/python3`
after installing into another Python. A pip upgrade is not required for these
commands. No packages are installed into the system Python.

Verification checks display divisors exactly within the same NumPy version.
Across versions it allows float32 rounding differences (relative tolerance
8 × float32 epsilon); input, label, split, and PNG hashes still require exact matches.
