"""
Build the FF100_SizeBM returns panel from the raw Kenneth French daily CSV.

Unlike the price->returns datasets (handled by preprocess_rmvp.py), the Fama-French file
already contains returns; the work here is *parsing* the right section and rescaling.

Source : "100 Portfolios Formed on Size and Book-to-Market (10x10) [Daily]", value-weighted.
         https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
         (ftp/100_Portfolios_10x10_Daily_CSV.zip)
Raw in  : datasets/raw/100_Portfolios_10x10_Daily.csv
Out     : datasets/FF100_SizeBM.xlsx   (header-less, days x 100 daily returns, decimal)

The CSV stacks FOUR sections (only the first is used):
    Average Value Weighted Returns -- Daily   <-- we take THIS (returns, in percent)
    Average Equal Weighted Returns -- Daily
    Number of Firms in Portfolios             (counts, e.g. 157 -- NOT returns)
    Average Firm Size

Process: extract the VW-daily section -> keep dates >= START -> drop any day that has a
missing value (-99.99/-999, i.e. require all 100 portfolios populated; this makes the
effective start 2004-07-01) -> percent/100 -> save. Reproduces datasets/FF100_SizeBM.xlsx.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "datasets" / "raw" / "100_Portfolios_10x10_Daily.csv"
OUT = HERE / "datasets" / "FF100_SizeBM.xlsx"
START = 20000101          # keep dates on/after this (effective start 2004-07-01 after drops)
MISSING = -99.98          # values <= this are Fama-French missing codes (-99.99 / -999)


def build():
    lines = RAW.read_text(encoding="latin-1").split("\n")

    # 1) locate the value-weighted daily-returns section header
    s = next(i for i, l in enumerate(lines)
             if "Average Value Weighted Returns" in l and "Daily" in l)
    # 2) next line = the 100 portfolio names
    names = [c.strip() for c in lines[s + 1].split(",") if c.strip()]

    rows = []
    for l in lines[s + 2:]:
        p = l.split(",")
        if not p[0].strip().isdigit():        # 3) blank / next-section header -> stop
            break
        if int(p[0].strip()) < START:         # 4) date filter
            continue
        vals = [float(x) for x in p[1:1 + len(names)]]
        if min(vals) <= MISSING:              # 5) drop any day with a missing portfolio
            continue
        rows.append(vals)

    R = np.array(rows) / 100.0                 # 6) percent -> decimal
    pd.DataFrame(R).to_excel(OUT, header=False, index=False)
    print(f"section '{lines[s].strip()}' at line {s}, {len(names)} portfolios")
    print(f"-> {OUT.relative_to(HERE)}: {R.shape[0]} days x {R.shape[1]} portfolios")


if __name__ == "__main__":
    build()
