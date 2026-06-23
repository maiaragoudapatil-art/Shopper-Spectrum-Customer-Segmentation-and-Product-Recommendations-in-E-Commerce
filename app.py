"""
🛒 Shopper Spectrum — Streamlit App
Modules:
  1. Product Recommendation (Collaborative Filtering)
  2. Customer Segmentation Predictor (KMeans RFM)

Run with:
    streamlit run app.py

Requires in same folder:
    kmeans_model.pkl, scaler.pkl, segment_labels.pkl,
    cosine_sim_df.pkl, product_list.pkl
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }

    /* Page title */
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #4C9BE8, #9B4CE8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: #888; font-size: 1rem; margin-bottom: 2rem; }

    /* Section headers */
    .section-header {
        font-size: 1.5rem; font-weight: 700; color: #ffffff;
        border-left: 4px solid #4C9BE8;
        padding-left: 12px; margin: 1.5rem 0 1rem 0;
    }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.2s;
    }
    .rec-card:hover { border-color: #4C9BE8; }
    .rec-rank {
        font-size: 1.5rem; font-weight: 800;
        color: #4C9BE8; min-width: 32px;
    }
    .rec-name { font-size: 0.95rem; color: #e0e0e0; font-weight: 500; }
    .rec-score { margin-left: auto; font-size: 0.8rem;
                 color: #888; background: #1a1d2e;
                 padding: 3px 8px; border-radius: 20px; }

    /* Segment badge */
    .seg-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 30px;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    .seg-High-Value  { background: #1a3d2b; color: #2ECC71; border: 2px solid #2ECC71; }
    .seg-Regular     { background: #1a2a3d; color: #3498DB; border: 2px solid #3498DB; }
    .seg-Occasional  { background: #3d2a1a; color: #E67E22; border: 2px solid #E67E22; }
    .seg-At-Risk     { background: #3d1a1a; color: #E74C3C; border: 2px solid #E74C3C; }

    /* Metric cards */
    .metric-row { display: flex; gap: 12px; margin: 1rem 0; }
    .metric-card {
        flex: 1; background: #1e2235;
        border: 1px solid #2e3250; border-radius: 10px;
        padding: 14px; text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #4C9BE8; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

    /* Input labels */
    label { color: #cccccc !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #4C9BE8, #9B4CE8);
        color: white; border: none;
        border-radius: 10px; font-size: 1rem;
        font-weight: 700; padding: 0.6rem 2rem;
        width: 100%; cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Divider */
    hr { border-color: #2a2d3e; }

    /* Info box */
    .info-box {
        background: #1a2235; border: 1px solid #2a3550;
        border-radius: 10px; padding: 14px 18px;
        font-size: 0.88rem; color: #aaa; margin: 1rem 0;
    }
    .info-box b { color: #4C9BE8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS  (cached so only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl',   'rb') as f: km      = pickle.load(f)
    with open('scaler.pkl',         'rb') as f: sc      = pickle.load(f)
    with open('segment_labels.pkl', 'rb') as f: seg_map = pickle.load(f)
    with open('cosine_sim_df.pkl',  'rb') as f: sim_df  = pickle.load(f)
    with open('product_list.pkl',   'rb') as f: prod_ls = pickle.load(f)
    return km, sc, seg_map, sim_df, prod_ls

km_model, scaler, segment_labels, cosine_sim_df, product_list = load_models()


# ─────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Home", "🎯 Product Recommendations", "👤 Customer Segmentation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#666; line-height:1.7'>
    <b style='color:#4C9BE8'>Models</b><br>
    KMeans (K=4)<br>
    Item-based Collaborative Filtering<br>
    Cosine Similarity<br><br>
    <b style='color:#4C9BE8'>Dataset</b><br>
    Online Retail 2022–2023<br>
    ~541K transactions<br>
    38 countries
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Customer Segmentation & Product Recommendation Engine</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in zip(
        [col1, col2, col3, col4],
        ["541K", "4,338", "4", f"{len(product_list)}"],
        ["Transactions", "Customers", "Segments", "Products Indexed"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">🎯 Product Recommendations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Uses <b>item-based collaborative filtering</b> with cosine similarity.<br>
        Enter any product name → get <b>5 most similar products</b> based on shared purchase patterns.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">👤 Customer Segmentation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Uses <b>KMeans clustering on RFM scores</b>.<br>
        Enter Recency, Frequency, Monetary → instantly predict which segment a customer belongs to:
        <b>High-Value</b>, <b>Regular</b>, <b>Occasional</b>, or <b>At-Risk</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Segment Profiles</div>', unsafe_allow_html=True)

    seg_info = [
        ("🏆", "High-Value",  "#2ECC71", "Recent, frequent, high spend. Your best customers — reward & retain."),
        ("⭐", "Regular",     "#3498DB", "Steady buyers, average spend. Nurture with loyalty programmes."),
        ("🔄", "Occasional",  "#E67E22", "Infrequent, low spend. Win back with targeted offers."),
        ("⚠️",  "At-Risk",    "#E74C3C", "Haven't bought in a long time. Urgent re-engagement needed."),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, name, color, desc) in zip([c1, c2, c3, c4], seg_info):
        with col:
            st.markdown(f"""
            <div style='background:#1e2235; border:1px solid {color}40;
                        border-radius:12px; padding:16px; text-align:center; height:160px'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='color:{color}; font-weight:800; font-size:1rem; margin:8px 0'>{name}</div>
                <div style='color:#888; font-size:0.8rem'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRODUCT RECOMMENDATIONS PAGE
# ─────────────────────────────────────────────
elif page == "🎯 Product Recommendations":
    st.markdown('<div class="main-title">🎯 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Item-based Collaborative Filtering · Cosine Similarity</div>', unsafe_allow_html=True)

    # Search input
    search = st.text_input(
        "🔍 Enter a product name (partial match works)",
        placeholder="e.g. CANDLE, MUG, BAG, HEART..."
    )

    # Dropdown of matching products
    if search:
        matches = [p for p in product_list if search.upper() in p.upper()]
        if matches:
            selected = st.selectbox("Select the exact product", matches)
        else:
            st.warning(f'No product found matching "{search}". Try a shorter keyword.')
            selected = None
    else:
        selected = st.selectbox("Or pick a product from the list", product_list)

    col_btn, col_gap = st.columns([1, 3])
    with col_btn:
        run = st.button("Get Recommendations ✨")

    if run and selected:
        st.markdown(f"**Recommendations similar to:** `{selected}`")
        st.markdown("---")

        sim_scores = (
            cosine_sim_df[selected]
            .drop(index=selected)
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        sim_scores.columns = ['Product', 'Score']

        for i, row in sim_scores.iterrows():
            pct = int(row['Score'] * 100)
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{i+1}</div>
                <div>
                    <div class="rec-name">{row['Product']}</div>
                    <div style='margin-top:6px; background:#0f1117;
                                border-radius:6px; height:6px; width:200px; overflow:hidden'>
                        <div style='height:100%; width:{pct}%;
                                    background:linear-gradient(90deg,#4C9BE8,#9B4CE8)'></div>
                    </div>
                </div>
                <div class="rec-score">{row['Score']:.4f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style='margin-top:1.5rem'>
        <b>How it works:</b> The similarity score (0–1) represents how often these products
        are bought by the same customers. A score of 1.0 means they are always purchased together.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTATION PAGE
# ─────────────────────────────────────────────
elif page == "👤 Customer Segmentation":
    st.markdown('<div class="main-title">👤 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">RFM Analysis · KMeans Clustering · Real-time Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter a customer's <b>Recency</b>, <b>Frequency</b>, and <b>Monetary</b> values
    to instantly predict which segment they belong to.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "📅 Recency (days since last order)",
            min_value=0, max_value=1000, value=30, step=1,
            help="0 = bought today. Higher = bought longer ago."
        )
    with col2:
        frequency = st.number_input(
            "🔁 Frequency (number of orders)",
            min_value=1, max_value=500, value=5, step=1,
            help="How many unique invoices the customer has."
        )
    with col3:
        monetary = st.number_input(
            "💷 Monetary (total £ spent)",
            min_value=0.0, max_value=500000.0, value=250.0, step=10.0,
            help="Total money spent across all orders."
        )

    col_btn2, _ = st.columns([1, 3])
    with col_btn2:
        predict = st.button("Predict Segment 🔮")

    if predict:
        # Scale input
        import math
        monetary_log = math.log1p(monetary)
        X_input = scaler.transform([[recency, frequency, monetary_log]])

        # Predict cluster
        cluster_id = int(km_model.predict(X_input)[0])
        segment    = segment_labels.get(cluster_id, f'Cluster {cluster_id}')

        seg_clean = segment.replace(' ', '-').replace('/', '-')

        st.markdown("---")
        st.markdown(f"### Predicted Segment")
        st.markdown(f'<div class="seg-badge seg-{seg_clean}">{segment}</div>', unsafe_allow_html=True)

        # Cluster centre stats (inverse transform for display)
        centre = km_model.cluster_centers_[cluster_id]
        avg_r  = round(scaler.mean_[0] + centre[0] * scaler.scale_[0], 0)
        avg_f  = round(scaler.mean_[1] + centre[1] * scaler.scale_[1], 1)
        avg_m  = round(math.expm1(scaler.mean_[2] + centre[2] * scaler.scale_[2]), 2)

        c1, c2, c3 = st.columns(3)
        for col, label, your_val, avg_val in zip(
            [c1, c2, c3],
            ["Recency (days)", "Frequency (orders)", "Monetary (£)"],
            [recency, frequency, monetary],
            [int(avg_r), avg_f, avg_m]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{your_val}</div>
                    <div class="metric-label">{label}<br>
                        <span style='color:#666'>Segment avg: {avg_val}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Segment-specific advice
        advice = {
            'High-Value' : ("🏆", "#2ECC71",
                "This is a top-tier customer. Reward with VIP loyalty perks, "
                "early access to new products, and personalised thank-you offers."),
            'Regular'    : ("⭐", "#3498DB",
                "A steady, reliable buyer. Encourage upselling with bundle deals "
                "and subscription programmes to move them towards High-Value."),
            'Occasional' : ("🔄", "#E67E22",
                "Buys infrequently. Run re-engagement campaigns with limited-time "
                "discounts or 'we miss you' emails to increase purchase frequency."),
            'At-Risk'    : ("⚠️",  "#E74C3C",
                "Hasn't purchased in a long time. Urgent win-back campaign needed — "
                "try a personalised discount or a 'last chance' email sequence."),
        }

        icon, color, tip = advice.get(segment, ("ℹ️", "#888", "No advice available."))

        st.markdown(f"""
        <div style='background:#1e2235; border-left:4px solid {color};
                    border-radius:10px; padding:16px 20px; margin-top:1.5rem'>
            <div style='font-size:1.1rem; font-weight:700; color:{color}'>
                {icon} Business Recommendation
            </div>
            <div style='color:#ccc; margin-top:8px; font-size:0.93rem'>{tip}</div>
        </div>""", unsafe_allow_html=True)
