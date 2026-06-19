"""
=========================================
TRAP CSV COORDINATE CLEANER
=========================================

Fixes:
- Decimal comma -> decimal point
- Converts coordinates to numeric
- Removes invalid coordinates
- Generates cleaned CSV

"""

import pandas as pd


INPUT_FILE = "traps_coordinates.csv"
OUTPUT_FILE = "traps_coordinates_clean.csv"


def clean_coordinates():

    print("Loading CSV...")

    df = pd.read_csv(INPUT_FILE, dtype=str)

    print("Rows loaded:", len(df))

    # remove quotes
    df["Latitud"] = df["Latitud"].str.replace('"', '')
    df["Longitud"] = df["Longitud"].str.replace('"', '')

    # replace decimal commas with dots
    df["Latitud"] = df["Latitud"].str.replace(",", ".", regex=False)
    df["Longitud"] = df["Longitud"].str.replace(",", ".", regex=False)

    # convert to numeric
    df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")

    # remove invalid rows
    df = df[
        (df["Latitud"].between(-90, 90)) &
        (df["Longitud"].between(-180, 180))
    ]

    print("Valid coordinates:", len(df))

    # save cleaned CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nClean CSV generated:")
    print(OUTPUT_FILE)


if __name__ == "__main__":

    clean_coordinates()