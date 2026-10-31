# Flood Prediction in Malawi — Zindi solution

A simple, reproducible baseline for the Zindi
[UNICEF Arm 2030 Vision #1: Flood Prediction in Malawi](https://zindi.world/competitions/2030-vision-flood-prediction-in-malawi)
challenge. It predicts the fraction of each ~1 km grid cell in southern Malawi
that is flooded in 2019, using terrain features and weekly rainfall.

**Competition result:** ranked **330 / 422**, best public submission RMSE
**0.1549**, Certificate of Participation.

<p align="center">
  <img src="assets/certificate.png" alt="Zindi certificate: ranked 330 out of 422" width="520">
</p>

> **Data is not included.** Zindi's competition rules prohibit publishing or
> redistributing the dataset. Download it from the competition page if you are
> an authorised participant. The files must stay out of git.
>
> **Not an early-warning system.** This is a research/education project, not an
> operational flood forecast. Do not use it for evacuation decisions.

## Quickstart

Requirements: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `Train.csv` and `SampleSubmission.csv` from the
[competition data page](https://zindi.world/competitions/2030-vision-flood-prediction-in-malawi/data)
and place them in `data/`:

```
data/
├── Train.csv
└── SampleSubmission.csv
```

Train and write a submission:

```bash
python train.py --model rf --cv 5
```

This prints cross-validation RMSE and writes
`submissions/submission.csv` with the required columns:

```
Square_ID,target_2019
```

## Results

Five-fold random cross-validation on the 16,466 training rows (seed `42`).
RMSE is the competition metric; lower is better.

| Model | Features | 5-fold CV RMSE |
|---|---|---:|
| `rf` | aggregated rainfall | 0.1264 |
| `rf` | raw weekly rainfall | 0.1264 |
| `hist_gb` | aggregated rainfall | **0.1200** |
| `rf --coordinates` | aggregated rainfall + X/Y | **0.0924** |

These are local CV scores, not leaderboard scores. The original submission
scored **0.1549** on Zindi. CV with coordinates is optimistic because nearby
cells appear in both train and validation; use it to compare models, not as a
deployment estimate.

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `rf` | `rf`, `extra_trees`, `hist_gb`, `tree`, or `ridge`. |
| `--cv` | `5` | Random K-fold count; `0` skips evaluation. |
| `--n-estimators` | `300` | Trees or boosting rounds. |
| `--coordinates` | off | Use `X`/`Y` as features. |
| `--raw-rainfall` | off | Use the 17 weekly values instead of aggregates. |
| `--train`, `--sample` | `data/Train.csv`, `data/SampleSubmission.csv` | Input files. |
| `--output` | `submissions/submission.csv` | Submission path. |
| `--id-col`, `--target-col` | `Square_ID`, `target_2015` | ID and label columns. |
| `--target-year`, `--predict-year` | `2015`, inferred | Rainfall events to use. |
| `--static-cols` | `elevation,LC_Type1_mode` | Extra model features. |
| `--categorical-cols` | `LC_Type1_mode` | Columns to one-hot encode. |
| `--seed` | `42` | Random seed. |

## Use another dataset

The script detects rainfall blocks from the dates in the column names, so it is
not tied to the Malawi files:

```bash
python train.py \
  --train my_data/Train.csv \
  --static-cols elevation,soil_type \
  --categorical-cols soil_type \
  --target-year 2015 \
  --predict-year 2019 \
  --output my_data/submission.csv
```

Expected format: one row per grid cell, an ID column, a `[0, 1]` target, static
columns, and at least two blocks of weekly columns named like
`precip 2019-01-20 - 2019-01-27`.

## How it works

1. Parse the weekly precipitation column names and group them into flood
   events (2015 for training, 2019 for prediction).
2. Summarise each event into total, mean, variability, maximum, recent sums,
   and wet-week counts. Optionally keep the raw weekly values.
3. Add elevation, land cover, and optionally coordinates.
4. Evaluate with random K-fold RMSE.
5. Refit on all training rows, clip predictions to `[0, 1]`, and write the
   submission in the sample-submission order.

## Notebook

The original competition notebook is in
[`notebooks/01_flood_prediction.ipynb`](notebooks/01_flood_prediction.ipynb).
Its outputs are cleared because they may contain competition data. The
maintained path is `train.py`.

## Notes

- The target is strongly zero-inflated: about 84% of cells have no flooding in
  2015.
- The 2015 and 2019 rainfall blocks have 17 weeks each. The test set reuses the
  same grid cells as training.
- The data was provided by the competition host; rainfall is from TRMM, elevation
  from SRTM, and land cover from MODIS.
- Local CV cannot tell us how well the 2015 relationship transfers to 2019. The
  original leaderboard score is the only external check.

## License

The code is released under the [MIT License](LICENSE). The competition data is
not included and remains subject to Zindi's rules and the competition host's
rights.
