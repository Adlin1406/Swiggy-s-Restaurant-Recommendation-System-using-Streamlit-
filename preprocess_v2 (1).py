"""
Swiggy Restaurant Recommendation System
Step 1: Data Cleaning and Preprocessing  (v2 - fixed)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import os

# ─────────────────────────────────────────────
# CONFIG  ← r"..." raw strings fix the backslash bug
# ─────────────────────────────────────────────
BASE_DIR          = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_Swiggy_4_GUVI"
RAW_DATA_PATH     = os.path.join(BASE_DIR, "swiggy.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
ENCODED_DATA_PATH = os.path.join(BASE_DIR, "encoded_data.csv")
ENCODER_PATH      = os.path.join(BASE_DIR, "encoder.pkl")


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)   # low_memory=False avoids DtypeWarning on large files
    print(f"[INFO] Shape    : {df.shape}")
    print(f"[INFO] Columns  : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[INFO] Starting data cleaning...")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows.")

    # Keep relevant columns only
    cols_needed = ['id', 'name', 'city', 'rating', 'rating_count', 'cost',
                   'cuisine', 'lic_no', 'link', 'address', 'menu']
    cols_present = [c for c in cols_needed if c in df.columns]
    df = df[cols_present].copy()

    # ── Rating ──────────────────────────────
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'].fillna(df['rating'].median(), inplace=True)
        df['rating'] = df['rating'].clip(0, 5)

    # ── Rating Count ─────────────────────────
    if 'rating_count' in df.columns:
        def parse_rating_count(val):
            val = str(val).strip().replace(',', '').replace('+', '')
            val = val.upper()
            if val.endswith('K'):
                try:
                    return float(val[:-1]) * 1000
                except ValueError:
                    return np.nan
            try:
                return float(val)
            except ValueError:
                return np.nan

        df['rating_count'] = df['rating_count'].apply(parse_rating_count)
        df['rating_count'].fillna(df['rating_count'].median(), inplace=True)

    # ── Cost ─────────────────────────────────
    if 'cost' in df.columns:
        df['cost'] = (
            df['cost']
            .astype(str)
            .str.replace('₹', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['cost'].fillna(df['cost'].median(), inplace=True)

    # ── City ─────────────────────────────────
    if 'city' in df.columns:
        df['city'] = df['city'].astype(str).str.strip().str.title()
        df = df[df['city'].notna() & (df['city'].str.lower() != 'nan')]

    # ── Cuisine ──────────────────────────────
    if 'cuisine' in df.columns:
        df['cuisine'] = (
            df['cuisine']
            .astype(str)
            .str.strip()
            .apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
            .replace('nan', 'Unknown')
            .replace('', 'Unknown')
        )

    # ── Name ─────────────────────────────────
    if 'name' in df.columns:
        df = df[df['name'].notna()]
        df['name'] = df['name'].astype(str).str.strip()

    # Final drop of remaining NaN rows in critical cols
    critical = [c for c in ['rating', 'cost', 'city', 'cuisine'] if c in df.columns]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    print(f"[INFO] Dropped {before - len(df)} rows with missing critical values.")

    # Reset index — CRITICAL for alignment with encoded_data
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Cleaned shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 3: ENCODE DATA
# ─────────────────────────────────────────────
def encode_data(df: pd.DataFrame):
    print("\n[INFO] Starting One-Hot Encoding...")

    cat_cols = [c for c in ['city', 'cuisine'] if c in df.columns]
    num_cols = [c for c in ['rating', 'rating_count', 'cost'] if c in df.columns]

    # Fit encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)

    # Build encoded dataframe (same index as cleaned_data)
    encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=df.index)

    # Add min-max normalised numerical columns
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        encoded_df[col] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0

    print(f"[INFO] Encoded shape: {encoded_df.shape}")
    return encoded_df, encoder


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    df_raw   = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)

    # Save cleaned data
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\n[SAVED] Cleaned data  → {CLEANED_DATA_PATH}")

    # Encode
    df_encoded, encoder = encode_data(df_clean)

    # Save encoded data
    df_encoded.to_csv(ENCODED_DATA_PATH, index=False)
    print(f"[SAVED] Encoded data  → {ENCODED_DATA_PATH}")

    # Save encoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"[SAVED] Encoder       → {ENCODER_PATH}")

    # Sanity check — index alignment
    assert len(df_clean) == len(df_encoded), \
        f"[ERROR] Index mismatch! cleaned={len(df_clean)}, encoded={len(df_encoded)}"
    print(f"\n[OK] Index alignment verified: {len(df_clean)} rows in both files.")

    print("\n─── Summary ───────────────────────────────")
    print(f"  Total restaurants : {len(df_clean)}")
    print(f"  Cities            : {df_clean['city'].nunique() if 'city' in df_clean.columns else 'N/A'}")
    print(f"  Cuisines          : {df_clean['cuisine'].nunique() if 'cuisine' in df_clean.columns else 'N/A'}")
    print(f"  Rating range      : {df_clean['rating'].min():.1f} – {df_clean['rating'].max():.1f}")
    print(f"  Cost range        : ₹{df_clean['cost'].min():.0f} – ₹{df_clean['cost'].max():.0f}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
