# WNBA Shot Atlas

An interactive study of where WNBA shots happen and how spatial models predict makes. Built with Python, JavaScript, HTML, and CSS using the 2025 season.

**[Explore the live app](https://kevinyamada.github.io/wnba/) · [Read the methods and audit](research/summary.md) · [Inspect saved results](research/results/)**

## What you can explore

- Shot density, make rates, and field goal scoring leaders across the half court.
- Team filters and 1, 2, or 4 foot cells, with keyboard selection.
- Distance baselines, IDW, kriging, Gaussian process, SPDE, and boosting comparisons.
- Temporal evaluation, buffered spatial validation, calibration, and court geometry checks.

The source contains 38,535 regular season attempts across 286 games; 38,441 fit in the displayed half court. The app uses saved aggregates and runs without a server backend or API key.

## Run the app locally

From this repository's root:

```bash
python -m http.server 8001
```

Open http://localhost:8001/. Serve the files over HTTP so the browser can load `assets/data.json`.

## Research and reproducibility

The [methods note](research/summary.md#reproduce-the-update) lists the Python 3.12 environment and full reproduction commands. Raw source files and event predictions are downloaded or generated locally and excluded from Git. Retained tables, source hashes, and verification outputs are included. Fitting the optional Bayesian model is separate from viewing the app.

These are exploratory results. Location summaries describe recorded attempts, and field goal points exclude free throws and future possession outcomes. Later evaluation periods have already been examined during development. This project does not yet estimate expected threat or causal shot-selection effects.

## Repository guide

| Path | Purpose |
| --- | --- |
| `index.html`, `style.css`, `app.js` | Interactive article and court explorer |
| `assets/data.json` | Published aggregates and model summaries |
| `research/` | Data preparation, geometry, models, evaluation, and methods |
| `research/results/` | Saved comparisons and verification outputs |
| `research/reference/` | Retained earlier analysis summaries |

## Source and publication

Kevin Yamada. Extracted from the existing [portfolio project at commit 1b4ddf6](https://github.com/kevinyamada/kevinyamada.github.io/tree/1b4ddf662053b9f4c331443fc5775788e319b4da/wnba) so the work has its own browsable repository. The live app remains hosted at its original portfolio URL. This repository is a standalone snapshot; changes here do not automatically update that deployment.

Data: [SportsDataverse / wehoop WNBA data](https://github.com/sportsdataverse/wehoop-wnba-data), sourced from ESPN. The [methods note](research/summary.md#sources) retains research references and data provenance. This is an independent project, unaffiliated with the WNBA or ESPN.
