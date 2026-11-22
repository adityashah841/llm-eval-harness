# benchmark-data

Run history written by the daily **Scheduled benchmark** workflow
(`.github/workflows/scheduled-benchmark.yml` on `main`). Code lives on `main`;
this branch only holds `metrics.db`.

Load it locally:

```bash
git fetch origin benchmark-data
mkdir -p data && git show origin/benchmark-data:metrics.db > data/metrics.db
```
