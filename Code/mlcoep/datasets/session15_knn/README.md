# NBA Player Statistics (Session 15)

**Purpose**: Offline copy of the data behind the NBA case study in `LaTeX/ml_knn_nba_case_study.tex`, so the session
runs without internet access.

## Files

| File | Rows | Columns | Used for |
|---|---|---|---|
| `nba_2013.csv` | 481 | 31 | Nearest neighbors by Euclidean distance, then KNN regression of points scored |

Columns: `player`, `pos`, `age`, `bref_team_id`, `g`, `gs`, `mp`, `fg`, `fga`, `fg.`, `x3p`, `x3pa`, `x3p.`, `x2p`, `x2pa`,
`x2p.`, `efg.`, `ft`, `fta`, `ft.`, `orb`, `drb`, `trb`, `ast`, `stl`, `blk`, `tov`, `pf`, `pts`, `season`, `season_end`.

The percentage columns (`fg.`, `x3p.`, `x2p.`, `ft.`) hold missing values for players who never attempted that shot
(94 empty cells across the columns the slides use). The slides fill them with 0.

## Usage

```python
import pandas

nba = pandas.read_csv('nba_2013.csv')
```

Runnable scripts: `sessions/session15_knn/nba_similar_players.py` (the case study) and
`sessions/session15_knn/knn_from_scratch.py`.

## Verification

Check that the file loads as 481 rows by 31 columns, and that `nba_similar_players.py` reproduces the case-study output on the slides.

## Provenance

Byte-identical to `Code/ml/data/nba_2013.csv` in this repository. The team column name (`bref_team_id`) points to
Basketball-Reference as the origin of the numbers; who assembled this particular CSV was not recorded.
