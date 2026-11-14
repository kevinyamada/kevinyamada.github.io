# Kevin Yamada — Portfolio

My project portfolio, built with HTML, CSS, and JavaScript and hosted on GitHub Pages.

**[Visit the portfolio](https://kevinyamada.github.io/) · [Explore the WNBA Shot Atlas](https://kevinyamada.github.io/wnba/) · [GitHub profile](https://github.com/kevinyamada)**

The site brings together my statistics, machine learning, AI safety, and data analysis projects, with links to code, reports, and interactive work. Team projects are credited in their source repositories.

## Run locally

```bash
python -m http.server 8001
```

Open http://localhost:8001/. Use an HTTP server to load the WNBA app's JSON data.

## Structure

- `index.html`: portfolio homepage and project filters.
- `wnba/`: live WNBA article, interactive court explorer, research code, and saved results.
- [WNBA source and research](wnba/): browse the app files and reproducible analysis.

Changes pushed to `main` are published through GitHub Pages. The `wnba/` folder in this repository remains the source of the live app.
