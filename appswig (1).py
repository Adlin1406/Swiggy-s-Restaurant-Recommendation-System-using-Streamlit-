"""
Swiggy Restaurant Recommendation System
Step 3: Streamlit Application  (v4 - memory fix)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR          = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_Swiggy_4_GUVI"
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
ENCODED_DATA_PATH = os.path.join(BASE_DIR, "encoded_data.csv")
ENCODER_PATH      = os.path.join(BASE_DIR, "encoder.pkl")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stButton>button {
        background: linear-gradient(135deg, #fc8019, #ff4500);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; padding: 0.6rem 2rem;
        font-size: 1rem; transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.88; }
    .restaurant-card {
        background: white; border-radius: 16px;
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #fc8019;
    }
    .restaurant-name { font-size: 1.1rem; font-weight: 700; color: #1a1a1a; }
    .restaurant-meta { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.75rem;
        font-weight: 600; margin-right: 6px;
    }
    .badge-rating { background: #e8f5e9; color: #2e7d32; }
    .badge-cost   { background: #fff3e0; color: #e65100; }
    .badge-score  { background: #e3f2fd; color: #1565c0; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #333;
        margin: 1.5rem 0 0.5rem;
        border-bottom: 2px solid #fc8019; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA — memory efficient
# @st.cache_resource avoids pickle serialization (no MemoryError)
# ─────────────────────────────────────────────
@st.cache_resource
def load_data():
    df_clean   = pd.read_csv(CLEANED_DATA_PATH)
    df_encoded = pd.read_csv(ENCODED_DATA_PATH, dtype=np.float32)  # float32 = half the RAM
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return df_clean, df_encoded, encoder


# ─────────────────────────────────────────────
# RECOMMENDATION HELPERS
# ─────────────────────────────────────────────
def build_user_vector(city, cuisine, rating, cost, rating_count,
                      encoder, df_encoded, df_clean):
    try:
        cat_encoded = encoder.transform([[city, cuisine]])
    except Exception:
        cat_encoded = np.zeros((1, len(encoder.get_feature_names_out())))

    def norm(val, col):
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        return (val - col_min) / (col_max - col_min) if col_max > col_min else 0.0

    num_cols = [c for c in ['rating', 'rating_count', 'cost'] if c in df_clean.columns]
    num_vals = {'rating': rating, 'rating_count': rating_count, 'cost': cost}
    num_part = np.array([[norm(num_vals[c], c) for c in num_cols]], dtype=np.float32)

    user_vec = np.hstack([cat_encoded.astype(np.float32), num_part])
    target_len = df_encoded.shape[1]
    if user_vec.shape[1] < target_len:
        user_vec = np.hstack([user_vec, np.zeros((1, target_len - user_vec.shape[1]), dtype=np.float32)])
    else:
        user_vec = user_vec[:, :target_len]
    return user_vec


def get_recommendations(city, cuisine, min_rating, max_cost, top_n,
                        df_clean, df_encoded, encoder):

    # ── STEP 1: Hard filter by city first (work on small subset) ──
    city_mask       = df_clean['city'].str.lower() == city.lower()
    filtered_clean  = df_clean[city_mask].copy()
    filtered_indices = filtered_clean.index.tolist()

    if len(filtered_clean) == 0:
        return pd.DataFrame()

    # Rating filter
    if 'rating' in filtered_clean.columns:
        r_mask = filtered_clean['rating'] >= min_rating
        if r_mask.sum() >= 3:
            filtered_clean   = filtered_clean[r_mask]
            filtered_indices = filtered_clean.index.tolist()

    # Cost filter
    if 'cost' in filtered_clean.columns:
        c_mask = filtered_clean['cost'] <= max_cost
        if c_mask.sum() >= 3:
            filtered_clean   = filtered_clean[c_mask]
            filtered_indices = filtered_clean.index.tolist()

    # ── STEP 2: Load only the filtered rows from encoded data ──
    # This avoids loading the full matrix into cosine_similarity
    df_encoded_filtered = df_encoded.loc[filtered_indices].values  # numpy array

    # ── STEP 3: Build user vector & compute similarity ──
    median_rc = df_clean['rating_count'].median()
    user_vec  = build_user_vector(city, cuisine, min_rating, max_cost,
                                   median_rc, encoder, df_encoded, df_clean)

    # Process in chunks to avoid RAM spike on large filtered sets
    CHUNK = 10000
    all_sims = np.empty(len(filtered_indices), dtype=np.float32)
    for start in range(0, len(filtered_indices), CHUNK):
        end = min(start + CHUNK, len(filtered_indices))
        all_sims[start:end] = cosine_similarity(
            user_vec, df_encoded_filtered[start:end]
        )[0]

    top_local   = np.argsort(all_sims)[::-1][:top_n]
    top_indices = [filtered_indices[i] for i in top_local]

    results = df_clean.loc[top_indices].copy()
    results['similarity_score'] = np.round(all_sims[top_local], 4)
    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("## 🍽️")
with col_title:
    st.markdown("# Swiggy Restaurant Recommender")
    st.caption("Discover your next favourite restaurant — powered by AI")

st.markdown("---")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df_clean, df_encoded, encoder = load_data()
except FileNotFoundError as e:
    st.error(f"⚠️ Data files not found: {e}\n\nPlease run `preprocess_v2.py` first.")
    st.stop()

cities   = sorted(df_clean['city'].dropna().unique().tolist())
cuisines = sorted(df_clean['cuisine'].dropna().unique().tolist())

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🍴 Total Restaurants", f"{len(df_clean):,}")
c2.metric("🏙️ Cities",            f"{df_clean['city'].nunique():,}")
c3.metric("🍜 Cuisine Types",     f"{df_clean['cuisine'].nunique():,}")
c4.metric("⭐ Avg Rating",        f"{df_clean['rating'].mean():.2f}")

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Your Preferences")

    city = st.selectbox("📍 City", options=cities)

    # Cuisine options filtered to selected city
    city_cuisines = sorted(
        df_clean[df_clean['city'].str.lower() == city.lower()]['cuisine']
        .dropna().unique().tolist()
    )
    cuisine = st.selectbox("🍜 Cuisine", options=city_cuisines if city_cuisines else cuisines)

    rating = st.slider("⭐ Minimum Rating", min_value=1.0, max_value=5.0,
                       value=3.5, step=0.1)
    cost = st.slider(
        "💰 Max Budget (₹ per person)",
        min_value=int(df_clean['cost'].min()),
        max_value=int(df_clean['cost'].max()),
        value=int(df_clean['cost'].max()),
        step=50,
    )
    top_n = st.number_input("🔢 Number of Recommendations",
                             min_value=1, max_value=30, value=10)

    st.markdown("---")
    city_count = (df_clean['city'].str.lower() == city.lower()).sum()
    st.info(f"📊 **{city_count:,}** restaurants in **{city}**")

    recommend_btn = st.button("🔍 Find Restaurants", use_container_width=True)


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if recommend_btn:
    with st.spinner(f"Finding best restaurants in {city}..."):
        try:
            results = get_recommendations(
                city=city, cuisine=cuisine,
                min_rating=rating, max_cost=cost,
                top_n=top_n,
                df_clean=df_clean, df_encoded=df_encoded,
                encoder=encoder,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if len(results) == 0:
        st.warning(f"No restaurants found in **{city}** matching your criteria. Try lowering the rating or increasing the budget.")
        st.stop()

    st.markdown(
        f"<div class='section-header'>🍽️ Top {len(results)} Recommendations in {city}</div>",
        unsafe_allow_html=True
    )

    for _, row in results.iterrows():
        name        = row.get('name', 'Unknown')
        address     = row.get('address', 'Address not available')
        cuisine_val = row.get('cuisine', '')
        r_rating    = row.get('rating', 'N/A')
        r_cost      = row.get('cost', 'N/A')
        r_score     = row.get('similarity_score', '')
        r_link      = str(row.get('link', ''))

        score_badge = f"<span class='badge badge-score'>Match: {r_score}</span>" if r_score != '' else ""
        link_html   = f" &nbsp;|&nbsp; <a href='{r_link}' target='_blank'>View on Swiggy 🔗</a>" \
                      if r_link and r_link not in ('nan', 'None', '') else ""

        st.markdown(f"""
        <div class='restaurant-card'>
            <div class='restaurant-name'>🍴 {name}</div>
            <div class='restaurant-meta'>
                📍 {address} &nbsp;|&nbsp; 🍜 {str(cuisine_val).title()}{link_html}
            </div>
            <div style='margin-top:8px'>
                <span class='badge badge-rating'>⭐ {r_rating}</span>
                <span class='badge badge-cost'>₹ {r_cost}</span>
                {score_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Recommendations as CSV",
        data=csv_bytes,
        file_name=f"recommendations_{city}_{cuisine}.csv",
        mime="text/csv",
    )

else:
    st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)
    display_cols = [c for c in ['name', 'city', 'cuisine', 'rating', 'cost', 'address']
                    if c in df_clean.columns]
    st.dataframe(df_clean[display_cols].head(20), use_container_width=True)
    st.info("👈 Set your preferences in the sidebar and click **Find Restaurants**!")
