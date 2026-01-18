"""
Apply post-processing to master_athlete_results.csv using utils.py:
- Convert 'mark' to seconds
- Clean 'mark' values
- Add 'Age' column
"""

import os
import pandas as pd
from utils import convert_to_seconds, calculate_age, clean_mark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "data", "2025_athlete_results.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "2025_athlete_results_clean.csv")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mark" in df.columns:
        df["Mark_Seconds"] = df.apply(
            lambda row: convert_to_seconds(row["mark"], row.get("discipline", "")), axis=1
        )
        df["mark_clean"] = df["mark"].apply(clean_mark)

    if "DOB" in df.columns:
        df["Age"] = df["DOB"].apply(calculate_age)

    return df

def main():
    print(f"📂 Loading {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)
    df_processed = process_data(df)

    print(f"💾 Saving processed data to {OUTPUT_CSV}")
    df_processed.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
