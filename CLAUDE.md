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

`geopatcher` has no dependency on any operator-graph composition library
— it's a standalone framework. Integration wrappers (`GridSampler`,
`ApplyToChips`, `Stitch`) for [geotoolz](https://github.com/jejjohnson/geotoolz)
or similar live in the consuming library.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
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

All implementation lives in `src/geopatcher/_src/`. The public API is
re-exported through `src/geopatcher/__init__.py`. The `_src` layer is
private and may be rearranged without notice.

Layout:

| Path                                  | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `src/geopatcher/_src/patch.py`        | `Patch` / `TemporalPatch` / `SpatioTemporalPatch` carriers |
| `src/geopatcher/_src/protocols.py`    | `Field` / `AsyncField` / `Domain` Protocols            |
| `src/geopatcher/_src/domains.py`      | `GridDomain` / `VectorDomain` / `PointDomain` (`RasterDomain` re-exported from `georeader`) |
| `src/geopatcher/_src/fields/`         | `RasterField` + extras-gated `XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField` |
| `src/geopatcher/_src/spatial/`        | `SpatialPatcher` + the four spatial axes               |
| `src/geopatcher/_src/time/`           | `TemporalPatcher` + the four temporal axes             |
| `src/geopatcher/_src/spatial_time.py` | `SpatioTemporalPatcher` (product / coupled coupling)   |

### Key directories

| Path | Purpose |
|------|---------|
| `src/geopatcher/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## Documentation Examples

Example notebooks live in `docs/notebooks/`. The tutorial notebooks
import `from geotoolz import Sequential, Lambda` (the operator-graph
bridge) to illustrate end-to-end pipelines — `geotoolz` is a soft
prerequisite for re-executing those notebooks. The committed `.ipynb`
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
