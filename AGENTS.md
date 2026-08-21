# AGENTS.md

Guidance for agents working in this repository.

## Overview

FaultForge simulates fault mitigation (error-correcting codes, bit-flip/stuck-at
fault injection) for PyTorch models. It's a hybrid project: performance-critical
bit-level encoding/fault logic lives in Rust, exposed to Python via PyO3, and the
experiment framework and model/dataset loading live in Python on top of it.

- `crates/` - Rust crates.
- `src/` - the `faultforge` library's Python source.
- `experiments/` - standalone experiment packages built on `faultforge`. Each
  pins an exact `faultforge` version rather than tracking `main`.

**After changing any Rust code, rebuild the extension(s) before running
Python tests**, otherwise Python will import the stale compiled `.so`:

```sh
.venv/bin/maturin develop
```
  
Do this in the directory of the package being tested if the package has a rust extension.

## Commands

`uv` by default creates a venv in `.venv`. Use the binaries in `.venv/bin`
directly rather than `uv run ...`.

### Rust

```sh
cargo clippy --workspace -- -D warnings   # lint (CI treats warnings as errors)
cargo test --workspace                    # test (default; don't assume nextest is installed)
cargo doc --workspace --no-deps --document-private-items  # doc check (CI runs this)
```

CI (`.github/workflows/rust.yml`) uses `cargo nextest` instead of `cargo test`;
use it if it's already installed, but don't require it.

### Python

```sh
.venv/bin/ty check
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/pytest
```

CI (`.github/workflows/python.yml`) runs the equivalent via `uv run`.

## Architecture

### Rust (`crates/`)

- `picker` - a Fisher-Yates-based random-permutation iterator, with support
  for resuming from a partial result.
- `memory` - bit-level buffer types and error-correcting-code encodings
  (see the crate-level doc comment in `crates/memory/src/lib.rs` for details)
  used to simulate protected memory and inject faults into it.
- `systolic` - a weight-stationary systolic-array simulator, with register
  fault injection and Rust-computed fault "lifts" for the fast torch-side path.
- `bindings` - the `faultforge._rust` extension module.
- `systolic_bindings` - an extension module for `experiments/systolic`.

### Python (`src/`, `experiments/`)

Top-level modules under `faultforge/*.py` are thin, documented re-export
shims; the real implementation lives in `faultforge/_internal/`. When changing
behavior, edit `_internal`; when adding a public symbol, re-export it from
the matching top-level shim. See `faultforge/__init__.py`'s module docstring
for a tour of the library's key parts (the experiment framework, encoding,
dataset/model loading, fault injection primitives).

### Testing conventions already in use

- Rust: `proptest` for property-based tests.
- Python: `hypothesis` for property-based tests.
