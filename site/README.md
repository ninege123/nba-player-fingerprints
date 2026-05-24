# NBA Player Fingerprints Site

This folder contains the rendered static MVP report and first interactive app published by GitHub Pages.

- Report: `/`
- App: `/app/`

To refresh it:

1. Regenerate processed data.
2. Render `reports/player_fingerprint_report.qmd`.
3. Copy the rendered HTML and support files into this folder with `player_fingerprint_report.html` renamed to `index.html`.
4. Regenerate the static app with `python -m nba_fingerprints.app.static_app`.
