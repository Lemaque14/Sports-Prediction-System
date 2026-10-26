# Sports Prediction System

This is a Streamlit multipage app with one predictor per sport. Each predictor is trained with PyCaret:

| Sport | Predicts | Model type | Page |
|---|---|---|---|
| IndyCar | Finishing order (pre- and post-qualifying) | Regression of normalized finish | `pages/indycar_page.py` |
| Liga MX | Win / draw / loss for a match | Classification | `pages/LigaMX_page.py` |
| UFC (men) | Each fighter's chance of winning in round 1–5 or by decision | Two classifiers multiplied | `pages/UFC_page.py` |

`sportssystem_homepage.py` is the entry point: `streamlit run sportssystem_homepage.py`.

The owner updates data and retrains in **Google Colab** (notebooks read from Google Drive). The app runs on Streamlit Cloud from this repo.

## Layout

```
sportssystem_homepage.py   entry point, links to the pages
pages/                     one Streamlit page per sport
features/                  feature code shared by the pages and the tests (indycar.py, ligamx.py, ufc.py)
tests/                     pytest; "replay" tests check the app's features against the datasets
<Sport>/datasets/          versioned CSVs (<Name>_vN.csv) and the notebook that builds them
<Sport>/MLModel_<Sport>.ipynb   training notebook
<Sport>/models/            PyCaret pipelines (.pkl, loaded with load_model(path_without_extension))
<Sport>/predictions/       backtests and model-comparison sheets
UFC/build_dataset.py       builds the UFC CSVs from Greco1899/scrape_ufc_stats at a pinned commit
```

Each page names the exact dataset and model versions it loads in constants near the top (`DATASET`, `MODEL`, …). To change a version, change those constants.

## Rules that matter

1. **Features use only what was known before the event.** The project has already been bitten by same-event leakage several times:
   - Liga MX Elo and head-to-head values on a match's second row included that match.
   - IndyCar track and team averages included the race being predicted.

   For Liga MX, both rows of a match get the same pre-match values. For IndyCar, a race's values join the history only after the whole race is filled in.
2. **The app must build the same features the model was trained on.** Each dataset row holds values from *before* that event, so the latest row is one event stale. The pages therefore rebuild the current state by replaying the full history with the code in `features/`.

   The replay tests check that the state just before event N equals the dataset's row for event N. If you change a feature in a notebook, change `features/` to match and keep `pytest` green.
3. **IDs stay stable between dataset versions.** Never re-derive IDs alphabetically (`cat.codes`), because one new name renumbers everyone after it and the trained model then reads each team as its neighbour.
   - Liga MX: `stable_ids` in `LigaMX_metrics.ipynb`, plus `features.ligamx.team_ids` in the app.
   - IndyCar: INDYCAR's own EventID and DriverID, and race folders named `<EventID>_<track>_<year>`.
4. **Data files are versioned, never overwritten.** Write `_v(N+1)`, then point the page and training notebook at it.
5. **Evaluate on a time-ordered holdout against a simple baseline** (Elo, home team wins, historical shares), never on a random split.
6. **Secrets never go in the repo** (it is public). The Liga MX scraper uses a ScraperAPI key: keep it in Colab Secrets (`from google.colab import userdata; userdata.get("SCRAPERAPI_KEY")`). Clear notebook outputs before committing.
7. Work on a branch and open a PR; don't push to `main`.

## Running and testing

```bash
pip install -r requirements.txt     # pinned; Python 3.11 (runtime.txt)
pytest                              # from the repo root (pytest.ini sets pythonpath)
streamlit run sportssystem_homepage.py
```

Pages can be tested headless with Streamlit's `AppTest`:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("sportssystem_homepage.py", default_timeout=300); at.run()
at.switch_page("pages/LigaMX_page.py").run()
at.selectbox(key="team_selected").select("Atlante").run()
at.button(key="ligamx_predict").click().run()
assert not at.exception and not at.error
```

Widget keys:
- IndyCar: `sim_track`, `sim_predict`
- Liga MX: `team_selected`, `opp_selected`, `ref_selected`, `ligamx_predict`
- UFC: `ufc_a`, `ufc_b`, `ufc_class`, `ufc_rounds`, `ufc_title`, `ufc_predict`

**CI** (`.github/workflows/python-app-test-sps.yml`) runs on PRs into `main` with Python 3.10. It checks for syntax errors and undefined names with flake8, then runs `pytest`. Exit code 5 means pytest found no tests, which is what `main` gives until `tests/` is merged.

### Environment gotchas

- **App environment:** pandas 2.1.4, numpy 1.26.4, PyCaret 3.3.2, Streamlit 1.43, pyarrow 19.0.1. pyarrow is pinned because newer versions need NumPy 2 and crash Streamlit on start against numpy 1.26.
- **Colab:** has pandas ≥ 2.2. The IndyCar metrics generator uses `groupby(...).apply(..., include_groups=False)`, which pandas 2.1 doesn't have, so run it in Colab (or any pandas ≥ 2.2 environment), not in the app environment.
- **Dates:** pandas 2.1 can't concatenate `datetime64[s]` with `datetime64[ns]`; cast to the frame's dtype first.
- **Tables:** `st.dataframe` needs one type per column (pyarrow), so cast mixed-type display columns to strings.
- **PyCaret `add_metric`:** needs a named function (not `functools.partial`). Multiclass metrics may receive string labels.

## Updating each sport

### IndyCar (after each race)
1. Type the race's results into the latest results file (`IndyCar_dataset_vN.csv` on Drive): finish, start, status, IDs. Use INDYCAR's EventID and DriverID.
2. In Colab, run `IndyCar/datasets/indycar_pdf_scrapper.ipynb`:
   - It downloads the race's PDFs from imscdn.com into `races/<EventID>_<track>_<year>/`.
   - It parses them into five CSVs in `scraped_data/`: pit stops, race metadata, cautions, lap positions and lap times.
3. Run `IndyCar/datasets/metrics_generator.ipynb`:
   - Set `INPUT_FILE` and `OUTPUT_FILE` first.
   - `check_results` stops on badly typed data, such as two dates for one EventID, positions that aren't 1..N, or a DriverID shared by two names.
   - `report_unmatched` lists scraped rows that didn't join.
4. Optionally retrain with `IndyCar/MLModel_IndyCar.ipynb`. Then point `pages/indycar_page.py` at the new dataset and models.

### Liga MX (each tournament, or as matches are played)
1. Scrape match results from FBref in Colab through ScraperAPI. The scraper notebook isn't in this repo; it saved `ligaMX_BaseDeDatos_v1.xlsx`.

   `LigaMX_metrics.ipynb` cell 1 reads `ligaMX_dataset_v1.xlsx` with `index_col=0`. Check the file name, and whether the file has an index column, before running.
2. Run `LigaMX/datasets/LigaMX_metrics.ipynb` with the previous dataset (`PREVIOUS_DATASET`) uploaded next to it, so IDs carry over. The last cell writes `OUTPUT_DATASET` in v6's exact format.

   Training uses every column it doesn't drop as a feature, so extra columns would break the app. Run on v6's raw columns, the notebook reproduces v6 exactly.
3. Retrain with `LigaMX/MLModel_LigaMX.ipynb`.
4. In `pages/LigaMX_page.py`, update:
   - `DATASET` and `MODEL`;
   - at the start of each tournament, `TORNEO, TEMPORADA` and `TEAMS_THIS_SEASON` (promoted, relegated and franchise-swapped clubs).

   A club with no history gets the next free ID and starts at Elo 1500. `Temporada` is the calendar year: Apertura 2026 is played July to December 2026.
5. `features.ligamx.missing_matches(df)` lists tournaments with fewer regular-season matches than a full round robin. Clausura 2020 is short by design: it was cancelled for COVID-19.

### UFC
- Set `SOURCE_COMMIT` in `UFC/build_dataset.py` to Greco1899/scrape_ufc_stats's latest commit and bump `VERSION`.
- Run `python UFC/build_dataset.py` from the repo root.
- Optionally rerun `UFC/MLModel_UFC.ipynb` (train 2008–2024, test 2025 on).
- Update the file names in `pages/UFC_page.py`.
- Women's divisions are left out on purpose for now.

## Open work (as of Oct 2026)

**Pull requests, in merge order:**
1. [#2](https://github.com/Lemaque14/Sports-Prediction-System/pull/2): leakage fixes, stale app features, rookies, `features/` and tests. Merge first.
2. [#3](https://github.com/Lemaque14/Sports-Prediction-System/pull/3): IndyCar generator and scraper fixes. Independent of #2. Its CI fails only because `main` has no tests yet; re-run it after #2.
3. [#4](https://github.com/Lemaque14/Sports-Prediction-System/pull/4): UFC page. Stacked on #2.
4. [#5](https://github.com/Lemaque14/Sports-Prediction-System/pull/5): Liga MX Apertura 2026 (Atlante) and prediction downloads. Stacked on #2.

After #2 merges, GitHub retargets #4 and #5 to `main`.

**Data fixes the owner needs to make:**
- **IndyCar results file**
  - 2014 Indy 500 (EventID 2988): 4 rows have TrackID 24; they should be 4.
  - Santino Ferrucci's 2026 races from 31 May to 23 Aug were typed with Katherine Legge's DriverID 4401; his is 4897.
  - Road America 2025 was typed with Thermal's EventID 6463. It is excluded (`EXCLUDED_RACES`) until its real EventID is confirmed.
- **IndyCar scraper:** run the one-time folder migration (`DRY_RUN = True` first), then generate v26 and retrain.
- **Liga MX re-scrape:** Apertura 2025 has 75 of 153 regular-season matches and Clausura 2026 has 96; Apertura 2014 is 1 short. Re-scrape them and add Apertura 2026. Then remove the `xfail` marker on `test_tournaments_are_complete` and retrain.
- **Liga MX club list:** confirm `TEAMS_THIS_SEASON` for Apertura 2026. Only Atlante replacing Mazatlán has been checked.
