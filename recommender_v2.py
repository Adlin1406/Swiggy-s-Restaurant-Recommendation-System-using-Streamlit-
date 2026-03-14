"""
Swiggy Restaurant Recommendation System
Step 2: Recommendation Engine  (v2 - fixed)
  - Cosine Similarity (primary)
  - K-Means Clustering (secondary / optional)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG  ← r"..." raw strings fix the backslash bug
# ─────────────────────────────────────────────
BASE_DIR          = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_Swiggy_4_GUVI"
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
ENCODED_DATA_PATH = os.path.join(BASE_DIR, "encoded_data.csv")
ENCODER_PATH      = os.path.join(BASE_DIR, "encoder.pkl")
KMEANS_PATH       = os.path.join(BASE_DIR, "kmeans_model.pkl")

N_CLUSTERS = 50   # tune based on dataset size


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts():
    df_clean   = pd.read_csv(CLEANED_DATA_PATH)
    df_encoded = pd.read_csv(ENCODED_DATA_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return df_clean, df_encoded, encoder


# ─────────────────────────────────────────────
# BUILD K-MEANS (call once, then reuse)
# ─────────────────────────────────────────────
def build_kmeans(df_encoded: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> KMeans:
    k = min(n_clusters, len(df_encoded))
    print(f"[INFO] Fitting K-Means with {k} clusters on {len(df_encoded)} rows...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_encoded.values)
    with open(KMEANS_PATH, 'wb') as f:
        pickle.dump(km, f)
    print(f"[SAVED] K-Means model → {KMEANS_PATH}")
    return km


def load_kmeans() -> KMeans:
    with open(KMEANS_PATH, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# BUILD USER VECTOR
# ─────────────────────────────────────────────
def build_user_vector(
    city: str,
    cuisine: str,
    rating: float,
    cost: float,
    rating_count: float,
    encoder: OneHotEncoder,
    df_encoded: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> np.ndarray:
    """
    Build a feature vector matching the encoded_data columns.
    Column order must match preprocess.py: [cat_features..., rating, rating_count, cost]
    """
    # One-hot encode city + cuisine
    try:
        cat_encoded = encoder.transform([[city, cuisine]])
    except Exception:
        cat_encoded = np.zeros((1, len(encoder.get_feature_names_out())))

    # Normalise numerics using same min/max from cleaned_data (matches preprocess.py)
    def norm(val, col):
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        return (val - col_min) / (col_max - col_min) if col_max > col_min else 0.0

    num_cols_present = [c for c in ['rating', 'rating_count', 'cost'] if c in df_clean.columns]
    num_vals = {'rating': rating, 'rating_count': rating_count, 'cost': cost}
    num_part = np.array([[norm(num_vals[c], c) for c in num_cols_present]])

    user_vec = np.hstack([cat_encoded, num_part])

    # Pad or trim to match encoded_data width exactly
    target_len = df_encoded.shape[1]
    if user_vec.shape[1] < target_len:
        pad = np.zeros((1, target_len - user_vec.shape[1]))
        user_vec = np.hstack([user_vec, pad])
    else:
        user_vec = user_vec[:, :target_len]

    return user_vec  # shape: (1, n_features)


# ─────────────────────────────────────────────
# COSINE SIMILARITY RECOMMENDATIONS
# ─────────────────────────────────────────────
def recommend_cosine(
    user_vec: np.ndarray,
    df_encoded: pd.DataFrame,
    df_clean: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top_n restaurants by cosine similarity."""
    sims = cosine_similarity(user_vec, df_encoded.values)[0]
    top_indices = np.argsort(sims)[::-1][:top_n]
    results = df_clean.iloc[top_indices].copy()
    results['similarity_score'] = np.round(sims[top_indices], 4)
    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# K-MEANS RECOMMENDATIONS
# ─────────────────────────────────────────────
def recommend_kmeans(
    user_vec: np.ndarray,
    km: KMeans,
    df_encoded: pd.DataFrame,
    df_clean: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find the user's cluster, then rank by cosine similarity within it."""
    cluster_label  = km.predict(user_vec)[0]
    cluster_indices = np.where(km.labels_ == cluster_label)[0]

    cluster_encoded = df_encoded.values[cluster_indices]
    sims            = cosine_similarity(user_vec, cluster_encoded)[0]
    top_local       = np.argsort(sims)[::-1][:top_n]
    top_global      = cluster_indices[top_local]

    results = df_clean.iloc[top_global].copy()
    results['similarity_score'] = np.round(sims[top_local], 4)
    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# UNIFIED RECOMMEND FUNCTION (used by Streamlit)
# ─────────────────────────────────────────────
def get_recommendations(
    city: str,
    cuisine: str,
    min_rating: float,
    max_cost: float,
    top_n: int = 10,
    method: str = "cosine",          # "cosine" | "kmeans"
    df_clean: pd.DataFrame = None,
    df_encoded: pd.DataFrame = None,
    encoder: OneHotEncoder = None,
    km: KMeans = None,
) -> pd.DataFrame:

    median_rc = df_clean['rating_count'].median()

    user_vec = build_user_vector(
        city=city, cuisine=cuisine,
        rating=min_rating, cost=max_cost,
        rating_count=median_rc,
        encoder=encoder,
        df_encoded=df_encoded,
        df_clean=df_clean,
    )

    # Fetch a larger pool first, then hard-filter
    pool_size = top_n * 5

    if method == "kmeans" and km is not None:
        results = recommend_kmeans(user_vec, km, df_encoded, df_clean, top_n=pool_size)
    else:
        results = recommend_cosine(user_vec, df_encoded, df_clean, top_n=pool_size)

    # ── Hard filters (only apply if enough results survive) ──
    def safe_filter(df, mask, min_keep=3):
        filtered = df[mask]
        return filtered if len(filtered) >= min_keep else df

    if 'city' in results.columns:
        results = safe_filter(results, results['city'].str.lower() == city.lower())

    if 'rating' in results.columns:
        results = safe_filter(results, results['rating'] >= min_rating)

    if 'cost' in results.columns:
        results = safe_filter(results, results['cost'] <= max_cost)

    return results.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# QUICK TEST (run directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_clean, df_encoded, encoder = load_artifacts()

    # Build (or reload) K-Means
    if os.path.exists(KMEANS_PATH):
        print("[INFO] Loading existing K-Means model...")
        km = load_kmeans()
    else:
        km = build_kmeans(df_encoded)

    # Use most common city & cuisine as test query
    city    = df_clean['city'].value_counts().index[0]
    cuisine = df_clean['cuisine'].value_counts().index[0]
    print(f"\n[TEST] city={city}, cuisine={cuisine}, min_rating=4.0, max_cost=500")

    for method in ["cosine", "kmeans"]:
        print(f"\n── Method: {method} ──────────────────────")
        recs = get_recommendations(
            city=city, cuisine=cuisine,
            min_rating=4.0, max_cost=500,
            top_n=5, method=method,
            df_clean=df_clean, df_encoded=df_encoded,
            encoder=encoder, km=km,
        )
        display_cols = [c for c in ['name', 'city', 'cuisine', 'rating', 'cost', 'similarity_score']
                        if c in recs.columns]
        print(recs[display_cols].to_string(index=False))
