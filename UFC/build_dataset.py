"""Build the UFC datasets from github.com/Greco1899/scrape_ufc_stats.

That repository scrapes ufcstats.com and refreshes its CSV files daily. This
script downloads them at a fixed commit (so the dataset can be rebuilt exactly),
keeps men's fights, and writes:

    UFC/datasets/UFC_fights_v<N>.csv    one row per fighter per fight
    UFC/datasets/UFC_fighters_v<N>.csv  height, reach, stance and date of birth

To refresh: set SOURCE_COMMIT to that repository's latest commit, bump VERSION,
run `python UFC/build_dataset.py` from the repository root, then point
pages/UFC_page.py at the new files.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.ufc import clean_raw  # noqa: E402

SOURCE_COMMIT = "3e2f3db5751c14b5d4ea52e41921982ca23b9b23"   # 2026-10-18 data refresh
SOURCE_URL = f"https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/{SOURCE_COMMIT}"
VERSION = 1
OUT_DIR = ROOT / "UFC" / "datasets"


def main():
    raw = {name: pd.read_csv(f"{SOURCE_URL}/ufc_{name}.csv")
           for name in ["event_details", "fight_results", "fight_stats", "fighter_tott"]}
    fights, fighters = clean_raw(raw["event_details"], raw["fight_results"],
                                 raw["fight_stats"], raw["fighter_tott"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fights.to_csv(OUT_DIR / f"UFC_fights_v{VERSION}.csv", index=False)
    fighters.to_csv(OUT_DIR / f"UFC_fighters_v{VERSION}.csv", index=False)
    print(f"{fights['FightID'].nunique()} men's fights from {fights['Date'].min():%Y-%m-%d} "
          f"to {fights['Date'].max():%Y-%m-%d}; {len(fighters)} fighters")


if __name__ == "__main__":
    main()
