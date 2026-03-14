import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

st.set_page_config(
    page_title="CRIMSON-India Dashboard",
    layout="wide",
    page_icon="🛡️",
)

# ─── Custom Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.article-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.article-card:hover { border-color: #6c63ff; }
.card-title { font-size: 1.05rem; font-weight: 600; color: #e0e0ff; margin-bottom: 4px; }
.card-meta  { font-size: 0.78rem; color: #8888aa; margin-bottom: 8px; }
.badge {
    display: inline-block;
    background: #3d3d6b;
    color: #a0a0ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin-right: 4px;
    margin-top: 4px;
}
.badge.crime  { background: #4a1e1e; color: #ff8080; }
.badge.accident { background: #3a2e10; color: #ffc060; }
.badge.cyber  { background: #102a3a; color: #60c0ff; }
.last-updated { font-size: 0.75rem; color: #6a6a8a; }
.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

CRIME_CATEGORIES = ['theft', 'assault', 'accident', 'drug_crime', 'cybercrime', 'non_crime']
BADGE_CLASSES = {
    'theft': 'crime', 'assault': 'crime',
    'accident': 'accident',
    'cybercrime': 'cyber',
    'drug_crime': 'crime', 'non_crime': '',
}


# ─── Data Helpers ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("data/labeled_news.csv"):
        return pd.read_csv("data/labeled_news.csv")
    return None


def fetch_live_news(max_per_feed: int = 20):
    """Run the full real-time pipeline: scrape → clean → classify → save."""
    from scraper.rss_scraper import RSSNewsScraper
    from preprocessing.text_cleaner import TextCleaner
    from utils.classifier_inference import classify_articles

    scraper = RSSNewsScraper()
    df_raw = scraper.scrape_to_csv("data/raw_news.csv", max_per_feed=max_per_feed)
    if df_raw.empty:
        return None, "❌ No articles fetched. Check your internet connection."

    cleaner = TextCleaner()
    df_raw['clean_text'] = df_raw['text'].apply(cleaner.clean_text)
    df_raw = df_raw[df_raw['clean_text'].str.len() > 0].reset_index(drop=True)

    df_classified = classify_articles(df_raw.copy())

    # Merge with existing data (keep newest, deduplicate by URL)
    if os.path.exists("data/labeled_news.csv"):
        df_old = pd.read_csv("data/labeled_news.csv")
        df_merged = pd.concat([df_classified, df_old], ignore_index=True)
        if 'url' in df_merged.columns:
            df_merged = df_merged.drop_duplicates(subset='url', keep='first')
        df_merged = df_merged.head(500)  # Cap at 500 rows
    else:
        df_merged = df_classified

    os.makedirs("data", exist_ok=True)
    df_merged.to_csv("data/labeled_news.csv", index=False)

    # Save fetch timestamp
    with open("data/.last_updated", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return df_merged, f"✅ Fetched {len(df_classified)} live articles from Indian news RSS feeds."


def get_last_updated():
    path = "data/.last_updated"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "Never"


def category_badges(row) -> str:
    labels = [c for c in CRIME_CATEGORIES if row.get(c, 0) == 1]
    if not labels:
        labels = ["unclassified"]
    badges = ""
    for lbl in labels:
        cls = BADGE_CLASSES.get(lbl, "")
        badges += f'<span class="badge {cls}">{lbl.replace("_", " ").title()}</span>'
    return badges


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=60)
st.sidebar.title("🛡️ CRIMSON-India")
st.sidebar.caption("Real-time Crime & Accident Monitor")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Live Data Controls")

articles_per_feed = st.sidebar.slider("Articles per source", 5, 30, 15, step=5)

if st.sidebar.button("🔄 Fetch Live News", use_container_width=True):
    with st.spinner("Scraping Indian news RSS feeds..."):
        df_new, msg = fetch_live_news(max_per_feed=articles_per_feed)
    load_data.clear()  # invalidate cache
    st.sidebar.success(msg) if "✅" in msg else st.sidebar.error(msg)
    st.rerun()

auto_refresh = st.sidebar.toggle("⏱️ Auto-refresh (5 min)", value=False)

last_updated = get_last_updated()
st.sidebar.markdown(f'<p class="last-updated">Last updated: {last_updated}</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📰 Live News Feed", "🗂️ Dataset Explorer", "📈 Trend Analysis", "🗺️ Geospatial (Demo)"])

st.sidebar.markdown("---")
st.sidebar.info("Powered by RSS feeds from NDTV, TOI, The Hindu, India Today & HT.")

# ─── Main Area ────────────────────────────────────────────────────────────────
df = load_data()

if page == "📰 Live News Feed":
    st.title("📰 Live Indian News Feed")
    st.markdown("Real-time articles scraped from Indian news RSS feeds, classified by crime category.")

    if df is None or 'source' not in df.columns:
        st.warning("No live data yet. Click **🔄 Fetch Live News** in the sidebar to get started.")
    else:
        # Filters
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sources = ["All Sources"] + sorted(df['source'].dropna().unique().tolist()) if 'source' in df.columns else ["All Sources"]
            selected_source = st.selectbox("Source", sources)
        with col2:
            selected_cat = st.selectbox("Category", ["All Categories"] + [c.replace("_", " ").title() for c in CRIME_CATEGORIES])
        with col3:
            n_show = st.number_input("Show", min_value=5, max_value=100, value=20, step=5)

        filtered = df.copy()
        if selected_source != "All Sources":
            filtered = filtered[filtered['source'] == selected_source]
        if selected_cat != "All Categories":
            cat_key = selected_cat.lower().replace(" ", "_")
            if cat_key in filtered.columns:
                filtered = filtered[filtered[cat_key] == 1]

        # Sort newest first
        if 'date' in filtered.columns:
            filtered = filtered.sort_values('date', ascending=False)

        st.markdown(f"**Showing {min(n_show, len(filtered))} of {len(filtered)} articles**")
        st.divider()

        shown = filtered.head(n_show)
        for _, row in shown.iterrows():
            title = row.get('title', 'Untitled')
            source = row.get('source', 'Unknown')
            date = row.get('date', '')
            url = row.get('url', '#')
            badges = category_badges(row)

            st.markdown(f"""
<div class="article-card">
  <div class="card-title"><a href="{url}" target="_blank" style="color:#e0e0ff;text-decoration:none;">{title}</a></div>
  <div class="card-meta">📰 {source} &nbsp;|&nbsp; 📅 {date}</div>
  <div>{badges}</div>
</div>
""", unsafe_allow_html=True)

elif page == "🗂️ Dataset Explorer":
    st.title("🗂️ Scraped & Classified News Dataset")
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", len(df))
        col2.metric("Sources", df['source'].nunique() if 'source' in df.columns else "—")
        col3.metric("Labelled", int(df[CRIME_CATEGORIES].sum().sum()) if all(c in df.columns for c in CRIME_CATEGORIES) else "—")

        st.subheader("Category Distribution")
        if all(c in df.columns for c in CRIME_CATEGORIES):
            counts = df[CRIME_CATEGORIES].sum().sort_values(ascending=False)
            st.bar_chart(counts)

        st.subheader("Raw Data")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Data not found. Click **🔄 Fetch Live News** in the sidebar or run `main.py --use-synthetic`.")

elif page == "📈 Trend Analysis":
    st.title("📈 Crime Trends Over Time")
    if os.path.exists("plots/crime_trends.png"):
        st.image("plots/crime_trends.png", caption="Monthly Detected Incident Trends", use_container_width=True)
    else:
        st.warning("Trend plot not found. Run the main pipeline to generate it.")

    if df is not None and all(c in df.columns for c in CRIME_CATEGORIES) and 'date' in df.columns:
        st.subheader("Interactive Trend (from current dataset)")
        df_t = df.copy()
        df_t['date'] = pd.to_datetime(df_t['date'], errors='coerce')
        df_t = df_t.dropna(subset=['date']).set_index('date')
        monthly = df_t[CRIME_CATEGORIES].resample('ME').sum()
        st.line_chart(monthly)

elif page == "🗺️ Geospatial (Demo)":
    st.title("🗺️ Geographic Crime Heatmap (Mocked Locations)")
    st.markdown("Extracted article locations mapped to approximate city coordinates across India.")

    if df is not None:
        import numpy as np
        cities = [
            (28.6139, 77.2090), (19.0760, 72.8777),
            (12.9716, 77.5946), (13.0827, 80.2707), (22.5726, 88.3639),
            (17.3850, 78.4867), (23.0225, 72.5714), (26.9124, 75.7873),
        ]
        rng = np.random.default_rng(seed=42)
        coords = []
        for _ in range(len(df)):
            lat, lon = cities[rng.integers(0, len(cities))]
            coords.append({'lat': lat + rng.normal(0, 0.5), 'lon': lon + rng.normal(0, 0.5)})
        st.map(pd.DataFrame(coords))
    else:
        st.warning("Data not found.")

# ─── Auto-refresh logic ──────────────────────────────────────────────────────
if auto_refresh:
    REFRESH_INTERVAL = 300  # 5 minutes
    placeholder = st.empty()
    for remaining in range(REFRESH_INTERVAL, 0, -1):
        placeholder.caption(f"⏱️ Auto-refreshing in {remaining}s...")
        time.sleep(1)
    placeholder.empty()
    fetch_live_news(max_per_feed=articles_per_feed)
    load_data.clear()
    st.rerun()
