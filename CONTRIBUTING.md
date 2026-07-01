# Contributing / project workflow

This is primarily a personal automotive ML research project, but the repo follows a lightweight workflow so changes are reproducible.

## Branch and commit workflow

For meaningful changes:

1. Start from `main`.
2. Make a focused change.
3. Run relevant checks.
4. Commit with a clear message.
5. Push to GitHub.

Suggested commit style:

```text
docs: clarify data collection guidelines
fix: improve Cars.com detail parser
feat: add dataset validation report
```

## Before pushing scraper changes

Run at least:

```bash
python3 -m pytest tests/ -q
```

If the change touches data cleaning or modeling, also run the relevant pipeline step on a small sample before committing.

## Before training models

1. Validate duplicate listing keys.
2. Check price and mileage parsing.
3. Review missing-value rates.
4. Confirm no target leakage features are included.
5. Save metrics and evaluation artifacts with the model outputs.

## Dataset handling

Raw CSV snapshots are kept for research reproducibility. Do not rewrite historical raw data unless correcting a clearly documented parsing error. Prefer appending new unique rows and documenting cleanup steps in code.

## GitHub Desktop usage

The repo is connected to GitHub and can be managed from GitHub Desktop. After local edits, review the diff, commit with a short descriptive message, and push to `origin/main`.
