import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fare Analytics",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  THEME / GLOBAL CSS
# ─────────────────────────────────────────────
NAVY   = "#1B2A4A"
TEAL   = "#2E8B7A"
SLATE  = "#4A6FA5"
LIGHT  = "#F0F4FA"
ACCENT = "#F4A261"
MUTED  = "#6B7C93"

st.markdown(f"""
<style>
  /* ── Base ── */
  html, body, [class*="css"] {{
      font-family: 'Segoe UI', sans-serif;
      background-color: {LIGHT};
  }}
  /* ── Hero header ── */
  .hero-box {{
      background: linear-gradient(135deg, {NAVY} 0%, {SLATE} 60%, {TEAL} 100%);
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 24px;
  }}
  .hero-title {{
      color: #ffffff;
      font-size: 2.1rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      margin: 0;
  }}
  .hero-sub {{
      color: #c8d8f0;
      font-size: 0.95rem;
      margin-top: 4px;
  }}
  /* ── KPI Cards ── */
  .kpi-card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 18px 20px;
      border-left: 5px solid {TEAL};
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .kpi-label {{
      color: {MUTED};
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 4px;
  }}
  .kpi-value {{
      color: {NAVY};
      font-size: 1.65rem;
      font-weight: 700;
      line-height: 1.2;
  }}
  .kpi-delta {{
      color: {TEAL};
      font-size: 0.82rem;
      margin-top: 2px;
  }}
  /* ── Section headers ── */
  .section-header {{
      color: {NAVY};
      font-size: 1.1rem;
      font-weight: 600;
      border-bottom: 2px solid {TEAL};
      padding-bottom: 6px;
      margin: 28px 0 16px 0;
  }}
  /* ── Insight / Rec boxes ── */
  .insight-box {{
      background: #ffffff;
      border-radius: 10px;
      padding: 18px 22px;
      border-left: 4px solid {SLATE};
      margin-bottom: 14px;
      box-shadow: 0 1px 5px rgba(0,0,0,0.06);
  }}
  .rec-box {{
      background: #ffffff;
      border-radius: 10px;
      padding: 18px 22px;
      border-left: 4px solid {ACCENT};
      margin-bottom: 14px;
      box-shadow: 0 1px 5px rgba(0,0,0,0.06);
  }}
  /* ── Chart row spacer ── */
  .chart-spacer {{
      margin-top: 8px;
      margin-bottom: 8px;
  }}
  .tag {{
      display: inline-block;
      background: {TEAL};
      color: #fff;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      margin-bottom: 6px;
      letter-spacing: 0.5px;
  }}
  .tag-accent {{
      background: {ACCENT};
      color: {NAVY};
  }}
  /* ── Sidebar base ── */
  section[data-testid="stSidebar"] {{
      background-color: {NAVY} !important;
  }}
  section[data-testid="stSidebar"] * {{
      color: #d4e1f7 !important;
  }}
  /* ── Sidebar filter labels (multiselect + slider) ── */
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stSlider label {{
      color: #93b8e0 !important;
      font-size: 0.88rem !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.9px;
  }}
  /* ── Sidebar input/dropdown option text ── */
  section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
  section[data-testid="stSidebar"] .stMultiSelect input {{
      color: #e8f0fb !important;
      font-size: 0.92rem !important;
  }}
  /* ── Sidebar multiselect dropdown list ── */
  section[data-testid="stSidebar"] [data-baseweb="popover"] li {{
      color: {NAVY} !important;
      font-size: 0.9rem !important;

/* ── Sidebar multiselect tag pills ── */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
    background-color: {TEAL} !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 4px !important;
    padding: 0px 4px !important;
    margin: 0px 1px !important;
    min-height: 17px !important;
}}

section[data-testid="stSidebar"] span[data-baseweb="tag"] span {{
    color: #ffffff !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    line-height: 1 !important;
}}

section[data-testid="stSidebar"] span[data-baseweb="tag"] button svg {{
    fill: rgba(255,255,255,0.75) !important;
}}
  /* ── Sidebar slider track & thumb ── */
  section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
      background-color: {TEAL} !important;
      border: 2px solid #ffffff !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {{
      background: {TEAL} !important;
  }}
  /* ── Sidebar author footer ── */
  .sidebar-footer {{
      position: relative;
      margin-top: 28px;
      padding-top: 14px;
      border-top: 1px solid rgba(255,255,255,0.1);
      text-align: center;
  }}
  .sidebar-footer-text {{
      color: #6b8cbb !important;
      font-size: 0.72rem;
      letter-spacing: 0.6px;
      line-height: 1.6;
  }}
  .sidebar-footer-name {{
      color: #93b8e0 !important;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.4px;
  }}
  /* ── Tab footer branding strip ── */
  .tab-footer {{
      margin-top: 48px;
      padding: 14px 0 6px 0;
      border-top: 1px solid #e0e7ef;
      text-align: center;
      color: #b0bec5;
      font-size: 0.72rem;
      letter-spacing: 1.2px;
      text-transform: uppercase;
  }}
  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 6px;
      background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: #dce8f5;
      border-radius: 8px 8px 0 0;
      color: {NAVY};
      font-weight: 600;
      padding: 8px 22px;
  }}
  .stTabs [aria-selected="true"] {{
      background: {NAVY} !important;
      color: #ffffff !important;
  }}
  /* ── Metric overrides ── */
  div[data-testid="metric-container"] {{
      background: #ffffff;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  /* ── Plotly chart outer card wrapper ── */
  div[data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) {{
      background-color: #ffffff;
      border: 1px solid #e6e9ef;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(149,157,165,0.2);
      padding: 20px;
      margin-bottom: 20px;
  }}
  /* ── Plotly chart inner element ── */
  [data-testid="stPlotlyChart"] {{
      background: transparent;
      border-radius: 0;
      box-shadow: none;
      padding: 0;
      margin-bottom: 0;
  }}
  /* ── KPI card lift on hover ── */
  .kpi-card:hover {{
      box-shadow: 0 8px 28px rgba(0,0,0,0.18), 0 2px 6px rgba(0,0,0,0.10);
      transform: translateY(-2px);
      transition: box-shadow 0.18s ease, transform 0.18s ease;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# shadow
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"], [data-testid="stHorizontalBlock"] > div, div.stPlotlyChart {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  DATA LOADING & PIPELINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & processing flight data…")
def load_data():
    df = pd.read_csv("Flight Fare Prediction.csv")

    # ── Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Drop duplicates & unnamed index col
    df.drop(columns=[c for c in df.columns if "s_no" in c or "s no" in c or "unnamed" in c.lower()], errors="ignore", inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Fix dtypes
    df["price"]    = pd.to_numeric(df["price"],    errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["days_left"]= pd.to_numeric(df["days_left"],errors="coerce")

    # ── Drop rows with critical nulls
    df.dropna(subset=["price", "duration", "days_left", "airline", "source_city", "destination_city"], inplace=True)

    # ── Remove obviously incorrect fares (< ₹500)
    df = df[df["price"] >= 500]

    # ── Feature engineering
    dep_hour_map = {
        "Early_Morning": 5, "Morning": 9,
        "Afternoon": 13,   "Evening": 17,
        "Night": 21,       "Late_Night": 1,
    }
    df["departure_hour"] = df["departure_time"].map(dep_hour_map).fillna(9).astype(int)

    # Booking window buckets
    bins   = [0, 7, 14, 30, 60, 9999]
    labels = ["0-7 days", "8-14 days", "15-30 days", "31-60 days", "60+ days"]
    df["booking_window"] = pd.cut(df["days_left"], bins=bins, labels=labels, right=True)

    # Route
    df["route"] = df["source_city"] + " → " + df["destination_city"]

    # Duration bucket
    dur_bins   = [0, 2, 4, 8, 9999]
    dur_labels = ["< 2 hrs", "2-4 hrs", "4-8 hrs", "8+ hrs"]
    df["duration_bucket"] = pd.cut(df["duration"], bins=dur_bins, labels=dur_labels)

    # Stops numeric
    stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
    df["stops_num"] = df["stops"].map(stops_map).fillna(1).astype(int)

    return df

df = load_data()

# ─────────────────────────────────────────────
#  MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training fare prediction model…")
def train_model(data):
    features = ["airline", "source_city", "destination_city",
                "departure_time", "stops_num", "days_left",
                "duration", "class"]
    target = "price"

    df_m = data[features + [target]].copy()

    encoders = {}
    cat_cols = ["airline", "source_city", "destination_city", "departure_time", "class"]
    for col in cat_cols:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str))
        encoders[col] = le

    X = df_m[features]
    y = df_m[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    return model, encoders, mae, r2

model, encoders, model_mae, model_r2 = train_model(df)

# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
CHART_COLORS = [TEAL, SLATE, NAVY, ACCENT, "#6BAED6", "#74C476", "#FD8D3C"]

def styled_fig(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI", size=12, color=NAVY),
        margin=dict(l=52, r=32, t=62, b=52),
        title=dict(
            font=dict(size=15, color=NAVY, family="Segoe UI", weight="bold"),
            pad=dict(b=14),
            x=0.0,
            xref="paper",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e0e7ef",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=4,
            x=0,
            y=-0.22,
            xanchor="left",
            yanchor="top",
            orientation="h",
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Segoe UI"),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#e0e7ef",
        tickfont=dict(size=11),
        title_standoff=14,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef1f6",
        linecolor="#e0e7ef",
        tickfont=dict(size=11),
        title_standoff=14,
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:4px;'>✈ Filters</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#3a5278;margin:6px 0 14px 0;'>", unsafe_allow_html=True)

    airlines_all = sorted(df["airline"].unique().tolist())
    sel_airlines = st.multiselect("Airlines", airlines_all, default=airlines_all)

    classes_all = sorted(df["class"].unique().tolist())
    sel_class = st.multiselect("Cabin Class", classes_all, default=classes_all)

    days_min, days_max = int(df["days_left"].min()), int(df["days_left"].max())
    sel_days = st.slider("Days Left to Departure", days_min, days_max, (days_min, days_max))

    sources_all = sorted(df["source_city"].unique().tolist())
    sel_source = st.multiselect("Source City", sources_all, default=sources_all)

    st.markdown("<hr style='border-color:#3a5278;margin:14px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;color:#8aa8cc;'>Dataset: {len(df):,} records after cleaning</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top: 14px; text-align: center; color: #8aa8cc; font-size: 0.78rem; font-weight: 400; letter-spacing: 0.3px;'>
        Developed by <span style='color: #c8d8f0; font-weight: 500;'>Vineeta Pandey</span>
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters
mask = (
    df["airline"].isin(sel_airlines) &
    df["class"].isin(sel_class) &
    df["days_left"].between(sel_days[0], sel_days[1]) &
    df["source_city"].isin(sel_source)
)
fdf = df[mask].copy()

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-box">
  <p class="hero-title">✈&nbsp; Flight Fare Intelligence Dashboard</p>
  <p class="hero-sub">India Domestic Routes &nbsp;|&nbsp; {len(fdf):,} filtered records &nbsp;|&nbsp; 6 Airlines &nbsp;|&nbsp; Data-Driven Fare Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "📈 Price Analytics",
    "🗺️ Route & Airline",
    "💡 Insights & Recommendations",
    "🤖 Fare Predictor",
])

# ══════════════════════════════════════════════
#  TAB 1 — KPI OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    if fdf.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar.")
    else:
        # ── KPI row
        total_flights  = len(fdf)
        avg_fare       = fdf["price"].mean()
        max_fare       = fdf["price"].max()
        min_fare       = fdf["price"].min()
        total_airlines = fdf["airline"].nunique()
        top_route      = fdf["route"].value_counts().idxmax()
        top_airline    = fdf["airline"].value_counts().idxmax()
        avg_duration   = fdf["duration"].mean()

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Total Flights</div>
              <div class="kpi-value">{total_flights:,}</div>
              <div class="kpi-delta">↑ Filtered dataset</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Average Fare</div>
              <div class="kpi-value">₹{avg_fare:,.0f}</div>
              <div class="kpi-delta">Range ₹{min_fare:,.0f} – ₹{max_fare:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Max Fare</div>
              <div class="kpi-value">₹{max_fare:,.0f}</div>
              <div class="kpi-delta">{total_airlines} airlines in selection</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Most Popular Route</div>
              <div class="kpi-value" style="font-size:1.1rem">{top_route}</div>
              <div class="kpi-delta">Top airline: {top_airline}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3, gap="medium")
        with c5:
            economy_share = (fdf["class"] == "Economy").mean() * 100
            st.markdown(f"""<div class="kpi-card" style="border-left-color:{SLATE};">
              <div class="kpi-label">Economy Share</div>
              <div class="kpi-value">{economy_share:.1f}%</div>
              <div class="kpi-delta">Business: {100-economy_share:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c6:
            direct_pct = (fdf["stops_num"] == 0).mean() * 100
            st.markdown(f"""<div class="kpi-card" style="border-left-color:{SLATE};">
              <div class="kpi-label">Non-stop Flights</div>
              <div class="kpi-value">{direct_pct:.1f}%</div>
              <div class="kpi-delta">{100-direct_pct:.1f}% with stops</div>
            </div>""", unsafe_allow_html=True)
        with c7:
            median_days = fdf["days_left"].median()
            st.markdown(f"""<div class="kpi-card" style="border-left-color:{SLATE};">
              <div class="kpi-label">Median Days Left</div>
              <div class="kpi-value">{median_days:.0f} days</div>
              <div class="kpi-delta">Booking lead time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        # ── Fare distribution overview
        st.markdown('<div class="section-header">Fare Distribution Overview</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            fig_hist = px.histogram(
                fdf, x="price", nbins=60,
                color_discrete_sequence=[TEAL],
                labels={"price": "Fare (₹)", "count": "Flights"},
                title="Fare Frequency Distribution",
            )
            fig_hist.update_traces(marker_line_color="white", marker_line_width=0.5)
            styled_fig(fig_hist, height=420)
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            class_agg = (
                fdf.groupby("class")["price"]
                .mean()
                .reindex(["Economy", "Business"])
                .reset_index()
            )
            fig_class_bar = px.bar(
                class_agg, x="class", y="price",
                color="class",
                color_discrete_sequence=[TEAL, SLATE],
                labels={"price": "Avg Fare (₹)", "class": "Cabin Class"},
                title="Average Fare by Cabin Class",
            )
            fig_class_bar.update_traces(
                texttemplate="₹%{y:,.0f}",
                textposition="outside",
                textfont=dict(size=13, color=NAVY),
            )
            fig_class_bar.update_layout(showlegend=False)
            styled_fig(fig_class_bar, height=420)
            st.plotly_chart(fig_class_bar, use_container_width=True)

        st.markdown('<div class="tab-footer">✈ &nbsp; Flight Fare Intelligence Dashboard &nbsp;·&nbsp; India Domestic Routes &nbsp;·&nbsp; Vineeta Pandey &nbsp; ✈</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — PRICE ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    if fdf.empty:
        st.warning("No data matches the current filters.")
    else:
        # ── Row 1: Departure time combo chart
        st.markdown('<div class="section-header">Price by Departure Time</div>', unsafe_allow_html=True)
        dep_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
        dep_agg = (
            fdf.groupby("departure_time")["price"]
            .agg(["mean","median","count"])
            .reindex([d for d in dep_order if d in fdf["departure_time"].unique()])
            .reset_index()
        )
        dep_agg.columns = ["departure_time", "mean", "median", "count"]

        fig_dep = go.Figure()
        fig_dep.add_trace(go.Bar(
            x=dep_agg["departure_time"], y=dep_agg["mean"],
            name="Avg Fare", marker_color=TEAL,
            text=[f"₹{v:,.0f}" for v in dep_agg["mean"]],
            textposition="outside",
        ))
        fig_dep.add_trace(go.Scatter(
            x=dep_agg["departure_time"], y=dep_agg["median"],
            name="Median Fare", mode="lines+markers",
            line=dict(color=ACCENT, width=2, dash="dot"),
            marker=dict(size=7),
        ))
        fig_dep.update_layout(
            title="Average & Median Fare by Departure Time Slot",
            xaxis_title="Departure Slot",
            yaxis_title="Fare (₹)",
            yaxis_tickprefix="₹",
        )
        styled_fig(fig_dep, height=430)
        st.plotly_chart(fig_dep, use_container_width=True)

        # ── Row 2: Booking window + Scatter
        st.markdown('<div class="section-header">Price by Days Left (Advance Booking Gap)</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            days_agg = (
                fdf.groupby("booking_window", observed=True)["price"]
                .mean().reset_index()
            )
            fig_days = px.bar(
                days_agg, x="booking_window", y="price",
                color="price",
                color_continuous_scale=[[0, LIGHT], [0.5, TEAL], [1, NAVY]],
                labels={"price": "Avg Fare (₹)", "booking_window": "Booking Window"},
                title="Avg Fare by Booking Window",
            )
            fig_days.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
            fig_days.update_layout(coloraxis_showscale=False, showlegend=False)
            styled_fig(fig_days, height=420)
            st.plotly_chart(fig_days, use_container_width=True)

        with col_b:
            sample_size = min(4000, len(fdf))
            scatter_df = fdf.sample(sample_size, random_state=1)
            fig_scatter = px.scatter(
                scatter_df, x="days_left", y="price",
                color="airline",
                color_discrete_sequence=CHART_COLORS,
                opacity=0.45,
                labels={"days_left": "Days Left", "price": "Fare (₹)", "airline": "Airline"},
                title=f"Fare vs Days Left (sample {sample_size:,})",
            )
            styled_fig(fig_scatter, height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Row 3: Duration & Stops
        st.markdown('<div class="section-header">Price by Duration & Stops</div>', unsafe_allow_html=True)
        col_c, col_d = st.columns(2, gap="large")

        with col_c:
            dur_agg = (
                fdf.groupby("duration_bucket", observed=True)["price"]
                .mean().reset_index()
            )
            fig_dur = px.bar(
                dur_agg, x="duration_bucket", y="price",
                color_discrete_sequence=[SLATE],
                labels={"price": "Avg Fare (₹)", "duration_bucket": "Flight Duration"},
                title="Avg Fare by Flight Duration",
            )
            fig_dur.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
            fig_dur.update_layout(showlegend=False)
            styled_fig(fig_dur, height=420)
            st.plotly_chart(fig_dur, use_container_width=True)

        with col_d:
            stops_agg = (
                fdf.groupby(["stops", "class"])["price"]
                .mean().reset_index()
            )
            fig_stops = px.bar(
                stops_agg, x="stops", y="price",
                color="class",
                barmode="group",
                color_discrete_sequence=[TEAL, NAVY],
                labels={"price": "Avg Fare (₹)", "stops": "Number of Stops", "class": "Class"},
                title="Avg Fare by Stops & Cabin Class",
            )
            fig_stops.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
            styled_fig(fig_stops, height=420)
            st.plotly_chart(fig_stops, use_container_width=True)

        # ── Row 4: Rolling fare trend (full-width)
        st.markdown('<div class="section-header">Fare Trend — Days to Departure (Rolling Avg)</div>', unsafe_allow_html=True)
        days_trend = (
            fdf.groupby("days_left")["price"]
            .mean().reset_index()
            .sort_values("days_left")
        )
        days_trend["rolling_avg"] = days_trend["price"].rolling(window=5, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=days_trend["days_left"], y=days_trend["price"],
            name="Avg Fare", mode="lines",
            line=dict(color="#c8d8f0", width=1),
        ))
        fig_trend.add_trace(go.Scatter(
            x=days_trend["days_left"], y=days_trend["rolling_avg"],
            name="Smoothed Trend", mode="lines",
            line=dict(color=TEAL, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(46,139,122,0.08)",
        ))
        fig_trend.update_layout(
            title="How Fares Change with Days Left to Departure",
            xaxis_title="Days Left",
            yaxis_title="Avg Fare (₹)",
            yaxis_tickprefix="₹",
            xaxis=dict(autorange="reversed"),
        )
        styled_fig(fig_trend, height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown('<div class="tab-footer">✈ &nbsp; Flight Fare Intelligence Dashboard &nbsp;·&nbsp; India Domestic Routes &nbsp;·&nbsp; Vineeta Pandey &nbsp; ✈</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — ROUTE & AIRLINE
# ══════════════════════════════════════════════
with tab3:
    if fdf.empty:
        st.warning("No data matches the current filters.")
    else:
        # ── Price by Airline
        st.markdown('<div class="section-header">Price by Airline</div>', unsafe_allow_html=True)
        airline_agg = (
            fdf.groupby("airline")["price"]
            .agg(["mean","median","min","max","count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        airline_agg.columns = ["Airline","Avg Fare","Median","Min","Max","Flights"]

        col_l, col_r = st.columns([3, 2], gap="large")
        with col_l:
            fig_airline = px.bar(
                airline_agg, x="Airline", y="Avg Fare",
                color="Airline",
                color_discrete_sequence=CHART_COLORS,
                labels={"Avg Fare": "Avg Fare (₹)"},
                title="Average Fare by Airline",
            )
            fig_airline.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
            fig_airline.update_layout(showlegend=False)
            styled_fig(fig_airline, height=430)
            st.plotly_chart(fig_airline, use_container_width=True)

        with col_r:
            fig_vol = px.pie(
                airline_agg, names="Airline", values="Flights",
                color_discrete_sequence=CHART_COLORS,
                title="Flight Volume Share by Airline",
                hole=0.45,
            )
            fig_vol.update_traces(
                textposition="inside", textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Flights: %{value:,}<extra></extra>",
            )
            fig_vol.update_layout(
                legend=dict(
                    orientation="v",
                    x=1.02, y=0.5,
                    xanchor="left", yanchor="middle",
                    font=dict(size=11),
                ),
                margin=dict(l=52, r=120, t=56, b=52),
            )
            styled_fig(fig_vol, height=430)
            st.plotly_chart(fig_vol, use_container_width=True)

        # ── Route heatmap
        st.markdown('<div class="section-header">Route Price Heatmap</div>', unsafe_allow_html=True)
        route_pivot = (
            fdf.groupby(["source_city","destination_city"])["price"]
            .mean()
            .unstack(fill_value=0)
        )
        fig_heat = px.imshow(
            route_pivot,
            color_continuous_scale=[[0, LIGHT], [0.4, TEAL], [1, NAVY]],
            labels={"x": "Destination", "y": "Source", "color": "Avg Fare (₹)"},
            title="Average Fare Heatmap: Source → Destination",
        )
        fig_heat.update_traces(texttemplate="₹%{z:,.0f}")
        fig_heat.update_layout(
            coloraxis_colorbar=dict(
                tickprefix="₹",
                thickness=14,
                len=0.75,
                x=1.02,
            ),
            margin=dict(l=52, r=80, t=56, b=72),
            xaxis=dict(side="bottom", tickangle=-30),
        )
        styled_fig(fig_heat, height=460)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Airline × Class price breakdown
        st.markdown('<div class="section-header">Airline Fare Comparison — Economy vs Business</div>', unsafe_allow_html=True)
        al_class = (
            fdf.groupby(["airline","class"])["price"]
            .mean().reset_index()
        )
        fig_alc = px.bar(
            al_class, x="airline", y="price",
            color="class", barmode="group",
            color_discrete_sequence=[TEAL, NAVY],
            labels={"price": "Avg Fare (₹)", "airline": "Airline", "class": "Cabin"},
            title="Economy vs Business Fare by Airline",
        )
        fig_alc.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
        styled_fig(fig_alc, height=430)
        st.plotly_chart(fig_alc, use_container_width=True)

        # ── Top routes table
        st.markdown('<div class="section-header">Top 10 Routes by Volume</div>', unsafe_allow_html=True)
        top_routes = (
            fdf.groupby("route")
            .agg(Flights=("price","count"), Avg_Fare=("price","mean"), Min_Fare=("price","min"), Max_Fare=("price","max"))
            .reset_index()
            .sort_values("Flights", ascending=False)
            .head(10)
        )
        top_routes["Avg_Fare"] = top_routes["Avg_Fare"].apply(lambda x: f"₹{x:,.0f}")
        top_routes["Min_Fare"] = top_routes["Min_Fare"].apply(lambda x: f"₹{x:,.0f}")
        top_routes["Max_Fare"] = top_routes["Max_Fare"].apply(lambda x: f"₹{x:,.0f}")
        top_routes.columns = ["Route", "Flights", "Avg Fare", "Min Fare", "Max Fare"]
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        st.dataframe(top_routes, use_container_width=True, hide_index=True)

        st.markdown('<div class="tab-footer">✈ &nbsp; Flight Fare Intelligence Dashboard &nbsp;·&nbsp; India Domestic Routes &nbsp;·&nbsp; Vineeta Pandey &nbsp; ✈</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab4:
    if fdf.empty:
        st.warning("No data matches the current filters.")
    else:
        # Compute live stats for insights
        best_dep   = fdf.groupby("departure_time")["price"].mean().idxmin()
        worst_dep  = fdf.groupby("departure_time")["price"].mean().idxmax()
        early_avg  = fdf[fdf["days_left"] > 30]["price"].mean()
        late_avg   = fdf[fdf["days_left"] <= 7]["price"].mean()
        pct_diff   = ((late_avg - early_avg) / early_avg) * 100
        cheapest_al = fdf.groupby("airline")["price"].mean().idxmin()
        priciest_al = fdf.groupby("airline")["price"].mean().idxmax()
        nonstop_avg = fdf[fdf["stops_num"] == 0]["price"].mean()
        onestop_avg = fdf[fdf["stops_num"] == 1]["price"].mean()
        biz_avg     = fdf[fdf["class"] == "Business"]["price"].mean()
        eco_avg     = fdf[fdf["class"] == "Economy"]["price"].mean()

        st.markdown('<div class="section-header">🔍 Key Data Insights</div>', unsafe_allow_html=True)

        insights = [
            ("Advance Booking Saves Money",
             f"Flights booked more than 30 days in advance average <b>₹{early_avg:,.0f}</b>, "
             f"while last-minute bookings (≤7 days) spike to <b>₹{late_avg:,.0f}</b> — "
             f"a <b>{pct_diff:.0f}% premium</b> for late-booking travellers."),
            ("Departure Time Drives Price",
             f"<b>{best_dep}</b> departures are the most affordable slot on average, "
             f"while <b>{worst_dep}</b> departures carry the highest average fares — "
             f"a consistent pattern across all airlines and routes."),
            ("Non-stop vs Connecting",
             f"Non-stop flights average <b>₹{nonstop_avg:,.0f}</b> vs "
             f"<b>₹{onestop_avg:,.0f}</b> for one-stop flights. "
             f"Passengers pay a <b>convenience premium</b> for direct travel."),
            ("Airline Price Spread",
             f"<b>{cheapest_al}</b> offers the most competitive fares on average, "
             f"while <b>{priciest_al}</b> commands the highest prices — "
             f"reflecting brand positioning and service differentiation."),
            ("Business vs Economy Premium",
             f"Business class fares average <b>₹{biz_avg:,.0f}</b> — "
             f"roughly <b>{((biz_avg/eco_avg)-1)*100:.0f}%</b> more than Economy "
             f"(avg ₹{eco_avg:,.0f}), highlighting strong yield in premium cabins."),
        ]

        for title, body in insights:
            st.markdown(f"""
            <div class="insight-box">
              <div class="tag">INSIGHT</div>
              <b style="color:{NAVY};">{title}</b><br>
              <span style="color:#4a5568;font-size:0.9rem;">{body}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🚀 Fare Price Business Recommendations</div>', unsafe_allow_html=True)

        recs = [
            ("Dynamic Advance-Booking Pricing Strategy",
             "Data shows a <b>37% fare surge</b> for last-minute bookings. "
             "Airlines should implement a <b>tiered dynamic pricing ladder</b>: offer the deepest discounts "
             "(10–15%) for bookings 45+ days out to lock in early revenue and boost load factors, "
             "apply standard fares between 15–44 days, and activate a progressive surge multiplier "
             "(1.2×–2.0×) within 14 days of departure. Pair this with <b>early-bird email campaigns</b> "
             "to shift demand curves and reduce last-seat volatility.",
             "STRATEGY 1 — Dynamic Pricing"),
            ("Peak Departure-Hour Surcharge & Off-Peak Incentive Programme",
             "<b>Night</b> slots already command premium prices — airlines can formalise this "
             "as an explicit <b>peak-hour surcharge (₹500–₹1,500)</b> on Night flights, "
             "while offering <b>off-peak incentives</b> (₹300–₹700 discount) on Late_Night "
             "departures to stimulate demand for underutilised time slots. "
             "This smooths capacity utilisation, maximises per-seat yield on prime slots, "
             "and trains price-sensitive travellers to self-select into off-peak windows — "
             "reducing ground congestion and improving turnaround efficiency.",
             "STRATEGY 2 — Slot Pricing"),
        ]

        for title, body, tag in recs:
            st.markdown(f"""
            <div class="rec-box">
              <div class="tag tag-accent">{tag}</div>
              <b style="color:{NAVY};">{title}</b><br>
              <span style="color:#4a5568;font-size:0.9rem;">{body}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="tab-footer">✈ &nbsp; Flight Fare Intelligence Dashboard &nbsp;·&nbsp; India Domestic Routes &nbsp;·&nbsp; Vineeta Pandey &nbsp; ✈</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 5 — FARE PREDICTOR
# ══════════════════════════════════════════════
with tab5:
    st.markdown(f"""
    <div style="background:{NAVY};border-radius:10px;padding:16px 24px;margin-bottom:18px;">
      <span style="color:#fff;font-size:1.15rem;font-weight:700;">🤖 AI Fare Predictor</span>
      <span style="color:#8aa8cc;font-size:0.85rem;margin-left:12px;">
        Random Forest · MAE ₹{model_mae:,.0f} · R² {model_r2:.3f}
      </span>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        st.markdown('<div class="section-header">Enter Flight Details</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            p_airline = st.selectbox("Airline", sorted(df["airline"].unique()))
        with r1c2:
            p_class = st.selectbox("Cabin Class", ["Economy", "Business"])

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            p_source = st.selectbox("Source City", sorted(df["source_city"].unique()))
        with r2c2:
            p_dest = st.selectbox("Destination City", sorted(df["destination_city"].unique()))

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            p_dep = st.selectbox(
                "Departure Time",
                ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
            )
        with r3c2:
            p_stops = st.selectbox("Stops", ["zero (Non-stop)", "one (1 Stop)", "two_or_more (2+ Stops)"])
            stops_val = {"zero (Non-stop)": 0, "one (1 Stop)": 1, "two_or_more (2+ Stops)": 2}[p_stops]

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            p_days = st.slider("Days Left to Departure", 1, 49, 14)
        with r4c2:
            p_dur = st.slider("Flight Duration (hrs)", 1.0, 20.0, 2.5, step=0.25)

        predict_btn = st.button("✈  Predict Fare", use_container_width=True)

    with col_out:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            if p_source == p_dest:
                st.error("Source and destination cities must be different.")
            else:
                with st.spinner("Calculating fare estimate…"):
                    # Encode inputs
                    def safe_encode(le, val):
                        if val in le.classes_:
                            return le.transform([val])[0]
                        return 0

                    row = {
                        "airline":           safe_encode(encoders["airline"],          p_airline),
                        "source_city":       safe_encode(encoders["source_city"],       p_source),
                        "destination_city":  safe_encode(encoders["destination_city"],  p_dest),
                        "departure_time":    safe_encode(encoders["departure_time"],    p_dep),
                        "stops_num":         stops_val,
                        "days_left":         p_days,
                        "duration":          p_dur,
                        "class":             safe_encode(encoders["class"],             p_class),
                    }
                    input_df = pd.DataFrame([row])
                    pred_fare = model.predict(input_df)[0]

                    # Comparable fare from data
                    comp = df[
                        (df["airline"] == p_airline) &
                        (df["source_city"] == p_source) &
                        (df["destination_city"] == p_dest) &
                        (df["class"] == p_class)
                    ]["price"]
                    data_avg = comp.mean() if not comp.empty else None

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{TEAL},{SLATE});
                            border-radius:14px;padding:28px 24px;text-align:center;margin-top:12px;">
                  <div style="color:#c8f0e8;font-size:0.82rem;letter-spacing:1px;text-transform:uppercase;">
                    Estimated Fare
                  </div>
                  <div style="color:#ffffff;font-size:3rem;font-weight:800;line-height:1.1;margin:8px 0;">
                    ₹{pred_fare:,.0f}
                  </div>
                  <div style="color:#d0ede8;font-size:0.88rem;">
                    {p_airline} · {p_source} → {p_dest} · {p_class}
                  </div>
                  <div style="color:#d0ede8;font-size:0.82rem;margin-top:4px;">
                    {p_dep} departure · {stops_val} stop(s) · {p_days} days left · {p_dur:.2f} hrs
                  </div>
                </div>
                """, unsafe_allow_html=True)

                google_flights_url = (
                    f"https://www.google.com/travel/flights?q=Flights%20from%20"
                    f"{p_source}%20to%20{p_dest}"
                )
                st.markdown(f"""
                <a href="{google_flights_url}" target="_blank" style="
                    background-color:#d4edda;
                    color:#155724;
                    border:1px solid #c3e6cb;
                    border-radius:8px;
                    padding:12px 18px;
                    text-decoration:none;
                    display:block;
                    text-align:center;
                    font-weight:600;
                    margin-top:12px;
                    font-size:0.95rem;">
                    🌐 Check Live Fares on Google Flights
                </a>
                """, unsafe_allow_html=True)

                if data_avg:
                    delta_pct = ((pred_fare - data_avg) / data_avg) * 100
                    direction = "above" if delta_pct > 0 else "below"
                    st.markdown(f"""
                    <div style="background:#f0f9f6;border-radius:8px;padding:12px 16px;
                                margin-top:12px;border:1px solid #c0e4da;">
                      <span style="color:{MUTED};font-size:0.82rem;">
                        Historical avg for this route/airline/class:
                        <b style="color:{NAVY};">₹{data_avg:,.0f}</b>
                        &nbsp;|&nbsp; Prediction is
                        <b style="color:{'#c0392b' if delta_pct>0 else TEAL};">
                          {abs(delta_pct):.1f}% {direction}
                        </b> historical average
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{LIGHT};border-radius:12px;padding:32px 20px;
                        text-align:center;border:2px dashed #c8d8f0;margin-top:12px;">
              <div style="font-size:2.5rem;">✈</div>
              <div style="color:{MUTED};font-size:0.9rem;margin-top:8px;">
                Fill in the flight details and click<br><b>Predict Fare</b> to get an AI estimate.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model performance card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#ffffff;border-radius:10px;padding:14px 18px;
                    border:1px solid #e0e7ef;font-size:0.83rem;color:{MUTED};">
          <b style="color:{NAVY};">Model Performance</b>&nbsp;&nbsp;
          Random Forest (120 trees, depth 18)<br>
          <span style="color:{TEAL};">MAE: ₹{model_mae:,.0f}</span>&nbsp;&nbsp;|&nbsp;&nbsp;
          <span style="color:{SLATE};">R²: {model_r2:.4f}</span>&nbsp;&nbsp;|&nbsp;&nbsp;
          Trained on {len(df):,} records
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="tab-footer">✈ &nbsp; Flight Fare Intelligence Dashboard &nbsp;·&nbsp; India Domestic Routes &nbsp;·&nbsp; Vineeta Pandey &nbsp; ✈</div>', unsafe_allow_html=True)
