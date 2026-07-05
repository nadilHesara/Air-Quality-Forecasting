# Contributing

Thanks for your interest in this project! It's a small, friendly codebase and
contributions are welcome. Here's how to help.

## Set up

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt -r requirements-dev.txt
```

## Before you open a pull request

Please run these two checks — CI runs the same ones, so this saves a round trip:

```bash
ruff check .        # style + lint
pytest              # all tests (no network needed)
```

Both should pass. If you add a feature, add a test for it too.

## A few house rules

- **Keep the feature code in one place.** All model features live in
  `src/features.py`. It's used by both training and serving, so the model can't
  see different inputs in the two places. Don't copy feature logic elsewhere.
- **No data leakage.** Every feature for day *t* must only use data from day *t*
  or earlier. There's a test for this (`tests/test_features.py`) — keep it green.
- **Tests stay offline.** Mock the network (see `tests/conftest.py`); tests must
  not call the real Open-Meteo APIs.
- **Simple English in docs.** The README and comments aim to be readable by
  non-experts. Match that tone.

## Good first tasks

- Add a new city and share the results.
- Improve the dashboard (charts, styling).
- Add more air-quality categories or health tips.

## Reporting bugs

Open a GitHub issue with what you expected, what happened, and the steps to
reproduce it. Small, clear reports get fixed fastest.
