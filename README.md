# benchmark-data

Run history written by the daily **Scheduled benchmark** workflow
(`.github/workflows/scheduled-benchmark.yml` on `main`). Code lives on `main`;
this branch only holds `metrics.db`.

Load it locally:

```bash
git fetch origin benchmark-data
mkdir -p data && git show origin/benchmark-data:metrics.db > data/metrics.db
```

Runs recorded before methodology v2 are migrated automatically on first use and
tagged `eval_version` 1. They are kept for history but not compared with newer runs.
