# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`geopatcher` is the four-axis Patcher framework for geospatial fields:
split a field into local patches, run an operator per patch, stitch
local outputs back into a global field. Three patcher families
(`SpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`) compose
the four axes (Geometry, Sampler, Window, Aggregation) over a `Field`
protocol that adapts the backend substrate (raster, xarray, geopandas,
xvec, …). Built with Python 3.12+, uv, pytest, and MkDocs.

`geopatcher`'s core has no dependency on any operator-graph composition
library — it's a standalone framework. The optional
`geopatcher.integrations.pipekit` submodule (gated behind the `[pipekit]`
extra) provides `GridSampler` / `ApplyToChips` / `Stitch` wrappers that
plug a `SpatialPatcher` into a [pipekit](https://github.com/jejjohnson/pipekit)
`Sequential` pipeline. While `pipekit` is pre-PyPI, install with
`uv sync --extra pipekit` (or `uv pip install` — see the README's
"Pre-PyPI install" section). Plain `pip install 'geopatcher[pipekit]'`
will work once `pipekit` reaches PyPI.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups --all-extras) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/geopatcher
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_sampler.py -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/geopatcher/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/geopatcher  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/geopatcher/`. CI runs `ruff check .` which includes `tests/`.

## Architecture

### Package structure

The framework core lives in the private `src/geopatcher/_src/` layer,
which may be rearranged without notice. The public API is re-exported
through `src/geopatcher/__init__.py` and thin public alias modules
(`fields.py`, `spatial.py`, `time.py`, `matched.py`, `hooks.py`). A
small, deliberate integration layer also sits at the package top level
outside `_src`: `runners.py` (reference executors, `parallel_map`),
`dask.py` (delayed / bag helpers), `jax/` (`BatchedPatch` batching),
and `integrations/pipekit.py` (operator-graph bridge). Invariant: core
logic in `_src`, only thin aliases and extras-gated integrations at the
top level.

`_src` layout:

| Path                                  | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `src/geopatcher/_src/patch.py`        | `Patch` / `TemporalPatch` / `SpatioTemporalPatch` carriers |
| `src/geopatcher/_src/protocols.py`    | `Field` / `AsyncField` / `Domain` Protocols            |
| `src/geopatcher/_src/domains.py`      | `GridDomain` / `VectorDomain` / `PointDomain` (`RasterDomain` re-exported from `georeader`) |
| `src/geopatcher/_src/fields/`         | `RasterField` + extras-gated `XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField`, `DaskField`, `ObstoreCogField` |
| `src/geopatcher/_src/spatial/`        | `SpatialPatcher` + the four spatial axes               |
| `src/geopatcher/_src/time/`           | `TemporalPatcher` + the four temporal axes + stencils  |
| `src/geopatcher/_src/spatial_time.py` | `SpatioTemporalPatcher` (product / coupled coupling)   |
| `src/geopatcher/_src/matched/`        | `MatchedField` + matched carriers / patchers (multi-source) |
| `src/geopatcher/_src/config.py`       | `get_strict` / `set_strict` strictness toggle          |
| `src/geopatcher/_src/exceptions.py`   | `IncompleteScanConfiguration`                          |
| `src/geopatcher/_src/hooks.py`        | `PatcherHook` callback protocol + dispatch             |
| `src/geopatcher/_src/indexed.py`      | `IndexedPatchView` random-access Sequence wrapper      |
| `src/geopatcher/_src/journal.py`      | `PatchJournal` resumable-job journal                   |
| `src/geopatcher/_src/cache.py`        | `PatchCache` content-addressed on-disk patch cache     |
| `src/geopatcher/_src/prefetch.py`     | `prefetch_iterable` background prefetching             |
| `src/geopatcher/_src/stacking.py`     | `stack_patches`                                        |
| `src/geopatcher/_src/objstore.py`     | Pooled obstore clients for `ObstoreCogField`           |

### Key directories

| Path | Purpose |
|------|---------|
| `src/geopatcher/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## Documentation Examples

Example notebooks live in `docs/notebooks/`. The committed tutorial
(`patcher_lake_tahoe.ipynb`) imports only `import geopatcher as gp`
plus numpy / matplotlib / pystac-client / planetary-computer /
rioxarray — it has no `geotoolz` dependency; those extra libraries are
the soft prerequisites for re-executing it. The committed `.ipynb`
files are pre-executed; `mkdocs-jupyter` renders them with
`execute: false`.

Figures render inline via `plt.show()` — do **not** use `savefig` or
commit separate PNG files.

## Coding Conventions

- Google-style docstrings
- `dataclasses` or `attrs` for data containers
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
