# Kevin Yamada — Portfolio

My project portfolio, built with HTML and CSS and hosted on GitHub Pages.

**[Visit the portfolio](https://kevinyamada.github.io/) · [WNBA Shot Atlas](https://kevinyamada.github.io/wnba-shot-atlas/) · [Canadian Household Finance](https://kevinyamada.github.io/canadian-household-finance/) · [GitHub profile](https://github.com/kevinyamada)**

The site brings together my statistics, machine learning, AI safety, and data analysis projects, with links to code, reports, and interactive work. Team projects are credited in their source repositories.

## Featured interactive projects

- [WNBA Shot Atlas](https://github.com/kevinyamada/wnba-shot-atlas): recorded shot maps, Bayesian spatial models, uncertainty, and an event-based possession-value prototype.
- [Canadian Household Finance](https://github.com/kevinyamada/canadian-household-finance): a housing-readiness audit with a provincial map, purchase-cash scenarios, shareable figures, and a Power BI report.

Each project is maintained and published from its own repository.

## Run locally

```bash
python -m http.server 8001
```

Open http://localhost:8001/.

## Structure

- `index.html`: portfolio homepage and project links.
- `wnba/index.html`: redirects the former shot-chart address to the current standalone app. Older research files remain for existing references; current methods and results live in the [WNBA repository](https://github.com/kevinyamada/wnba-shot-atlas/tree/main/research).

Changes pushed to `main` are published through GitHub Pages.
