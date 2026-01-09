import pandas as pd
from pathlib import Path


def prices_to_returns(folder_in, folder_out):
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)
    folder_out.mkdir(exist_ok=True, parents=True)
    
    files = list(folder_in.glob("*.xlsx")) + list(folder_in.glob("*.xls"))
    print(f"Found {len(files)} files in {folder_in}")

    for file in files:
        print(f"Processing {file.name}...")
        prices = pd.read_excel(file, sheet_name="Assets", header=None)

        returns = prices.pct_change().iloc[1:]

        out_file = folder_out / f"{file.stem}_returns.xlsx"
        returns.to_excel(out_file, sheet_name="Assets", header=False, index=False)
        print(f"Saved to {out_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    #prices_to_returns(folder_in=base_dir / "Datasets_DEF", folder_out=base_dir / "datasets")
    prices_to_returns(folder_in=base_dir / "Datasets_GeneralizedDR", folder_out=base_dir / "datasets")