# Machine Sensor Logs (Session 4: Pandas)

**Purpose**: Hands-on dataset for the Session 4 Pandas walkthrough (`LaTeX/python_intro_pandas.tex`)
**Format**: CSV, 6 rows, 5 columns

## Columns

| Column | Type | Description |
|---|---|---|
| `machine_id` | string | `M1`, `M2`, or `M3` |
| `shift` | string | `Morning` or `Evening` |
| `temperature` | float | Sensor reading (deg C) |
| `vibration` | float | Sensor reading (mm/s) |
| `operator_notes` | string | Free-text note logged by the operator |

## Usage

Give students this file before class so they can run the deck's `pd.read_csv('machine_logs.csv')`
examples themselves, no typing required. Place it in the same folder as their notebook/script.

The numbers were reverse-engineered to exactly match every output already shown in the slide deck
(shape, `value_counts`, groupby averages, vibration flags, morning/evening split), so running the
deck's code against this file reproduces every printed result verbatim.
