import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------
# CONFIG & MODEL LOAD
# -----------------------------
st.set_page_config(
    layout="wide",
    page_title="RUL Prediction Dashboard",
    page_icon="🔧",
)

# Custom CSS for a cleaner industrial look
st.markdown("""
    <style>
        .metric-card {
            background: #0f1923;
            border: 1px solid #1e3a4f;
            border-radius: 8px;
            padding: 16px 20px;
        }
        .stMetric { padding: 0 !important; }
        div[data-testid="stMetricValue"] { font-size: 2rem !important; }
        .block-container { padding-top: 2rem !important; }
        .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the model once and keep it in memory."""
    return joblib.load("rul_model.pkl")


# Refined feature list
FEATURES = [
    'Casing (Choke) Pressure kPa', 'Pump 2 Stroke Rate 1/min',
    'Average Hookload kkgf', 'Rig Mode unitless',
    'Cement flowrate In (avg) m3/min', 'Tank volume (total) m3',
    'Lag Depth (MD) m', 'Hole depth (MD) m', 'Mud Flow Out % %',
    'Mud Density In g/cm3', 'Tank 5 volume m3', 'Pump 1 Stroke Rate 1/min',
    'Average Rotary Speed rpm', 'Tank 4 volume m3', 'Tank 14 volume m3',
    'Weight on Bit kkgf', 'Bit Depth (MD) m',
    'Average Standpipe Pressure kPa', 'Pump 3 Stroke Rate 1/min',
    'Mud Temperature In degC', 'Mud Density Out g/cm3',
    'Cement pump pressure (avg) kPa', 'Mud Temperature Out degC',
    'Tank 11 volume m3', 'Rate of Penetration m/h', 'Block Position m',
    'Tank 6 volume m3', 'Tank 10 volume m3', 'Active Tank Volume m3',
    'Active Tank Volume Change m3', 'Average Surface Torque kN.m',
    'Tank 12 volume m3', 'Tank 9 volume m3', 'Trip tank 2 volume m3',
    'Hole Depth (TVD) m', 'Trip tank 1 volume m3', 'Gas %',
    'Tank 2 volume m3', 'Mud Flow In L/min', 'Tank 7 volume m3',
    'Tank 8 volume m3', 'Tank 13 volume m3', 'Tank 3 volume m3',
    'Tank 1 volume m3'
]


# -----------------------------
# DATA PROCESSING
# -----------------------------
@st.cache_data(show_spinner="Processing data and running predictions...")
def get_processed_data(file_bytes: bytes) -> tuple[pd.DataFrame | None, list | None]:
    """
    Load, clean, and predict RUL from raw CSV bytes.

    Caches on file_bytes (immutable), so the heavy work only runs
    when a genuinely new file is uploaded.

    Returns:
        (processed_df, None)        on success
        (None, list_of_missing_cols) on failure
    """
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

    # --- Datetime parsing ---
    if "DateTime parsed" in df.columns:
        df["DateTime parsed"] = pd.to_datetime(
            df["DateTime parsed"], format="mixed", utc=True, errors="coerce"
        )
        # Drop rows where datetime couldn't be parsed
        df = df.dropna(subset=["DateTime parsed"])
        df["DateTime parsed"] = df["DateTime parsed"].dt.tz_convert(None).dt.floor("min")
    else:
        # Fallback: integer index as a synthetic time axis
        st.warning(
            "⚠️ 'DateTime parsed' column not found. "
            "Using row index as time axis — time filtering will be disabled."
        )
        df["DateTime parsed"] = pd.RangeIndex(start=0, stop=len(df))

    # --- Check for missing feature columns before cleaning ---
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, missing

    # --- Numeric cleaning ---
    for col in FEATURES:
        series = df[col]

        # 1. Fill NaN with column mean (fallback to 0 if all NaN)
        col_mean = series.mean()
        if pd.isna(col_mean):
            col_mean = 0.0
        series = series.fillna(col_mean)

        # 2. Replace negative values with mean of positive values (fallback to 0)
        pos_mean = series[series > 0].mean()
        if pd.isna(pos_mean):
            pos_mean = 0.0
        series = series.where(series >= 0, other=pos_mean)

        df[col] = series

    # --- Sort and predict ---
    df = df.sort_values("DateTime parsed").reset_index(drop=True)

    model = load_model()
    preds = model.predict(df[FEATURES])
    df["Predicted_RUL"] = np.clip(preds, 0, None)

    return df, None


def build_rul_chart(df: pd.DataFrame) -> go.Figure:
    """Build a styled Plotly RUL trend chart."""
    rul = df["Predicted_RUL"]

    # Colour-map: green → yellow → red based on RUL percentile
    rul_norm = (rul - rul.min()) / (rul.max() - rul.min() + 1e-9)

    fig = go.Figure()

    # Shaded area under the curve
    fig.add_trace(go.Scatter(
        x=df["DateTime parsed"],
        y=rul,
        fill="tozeroy",
        fillcolor="rgba(46, 134, 193, 0.08)",
        line=dict(color="#2E86C1", width=2),
        name="Predicted RUL",
        hovertemplate="<b>%{x}</b><br>RUL: %{y:.2f}<extra></extra>",
    ))

    # Critical threshold line at 10th percentile
    threshold = float(np.percentile(rul, 10))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#E74C3C",
        line_width=1.5,
        annotation_text=f"Low RUL threshold ({threshold:.1f})",
        annotation_position="top right",
        annotation_font_color="#E74C3C",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            title="Time",
            gridcolor="rgba(255,255,255,0.05)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title="Predicted RUL",
            gridcolor="rgba(255,255,255,0.05)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# -----------------------------
# USER INTERFACE
# -----------------------------
st.title("🔧 RUL Prediction Dashboard")
st.caption("Upload a sensor CSV to generate Remaining Useful Life predictions.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file above to get started.", icon="📊")
    st.stop()

# Read bytes once — this is what gets passed to (and hashed by) the cached function
file_bytes = uploaded_file.read()

processed_df, missing_cols = get_processed_data(file_bytes)

if processed_df is None:
    st.error(
        f"❌ Your CSV is missing **{len(missing_cols)}** required column(s):\n\n"
        + ", ".join(f"`{c}`" for c in missing_cols)
    )
    st.stop()

# -----------------------------
# TIME FILTER
# -----------------------------
has_real_datetime = pd.api.types.is_datetime64_any_dtype(processed_df["DateTime parsed"])

st.divider()
st.subheader("🕒 Time Filter")

if has_real_datetime:
    min_t = processed_df["DateTime parsed"].min().to_pydatetime()
    max_t = processed_df["DateTime parsed"].max().to_pydatetime()

    c1, c2 = st.columns(2)
    start_t = c1.datetime_input("Start Time", value=min_t, min_value=min_t, max_value=max_t)
    end_t = c2.datetime_input("End Time", value=max_t, min_value=min_t, max_value=max_t)

    if start_t > end_t:
        st.warning("⚠️ Start time cannot be later than end time.")
        st.stop()

    filtered_df = processed_df[
        (processed_df["DateTime parsed"] >= pd.to_datetime(start_t)) &
        (processed_df["DateTime parsed"] <= pd.to_datetime(end_t))
    ]
else:
    # Integer index fallback — use a row-range slider
    total_rows = len(processed_df)
    row_range = st.slider("Row range", 0, total_rows - 1, (0, total_rows - 1))
    filtered_df = processed_df.iloc[row_range[0]: row_range[1] + 1]

if filtered_df.empty:
    st.warning("⚠️ No data found in the selected range. Please adjust your filter.")
    st.stop()

# -----------------------------
# METRICS
# -----------------------------
st.divider()

rul_series = filtered_df["Predicted_RUL"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("🔻 Min RUL",  f"{rul_series.min():.2f}")
m2.metric("🔺 Max RUL",  f"{rul_series.max():.2f}")
m3.metric("📊 Avg RUL",  f"{rul_series.mean():.2f}")
m4.metric("📋 Rows",     f"{len(filtered_df):,}")

# -----------------------------
# CHART
# -----------------------------
st.subheader("📉 RUL Trend")
fig = build_rul_chart(filtered_df)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FEATURE TREND EXPLORER
# -----------------------------
st.divider()
st.subheader("🔍 Feature Trend Explorer")

available_features = [c for c in FEATURES if c in filtered_df.columns]
selected_features = st.multiselect(
    "Select features to plot",
    options=available_features,
    default=available_features[:2] if len(available_features) >= 2 else available_features,
    placeholder="Choose one or more features...",
)

if selected_features:
    normalize = st.toggle("Normalize to [0, 1] for comparison", value=False)

    feat_fig = go.Figure()
    for feat in selected_features:
        y = filtered_df[feat]
        if normalize:
            y_min, y_max = y.min(), y.max()
            y = (y - y_min) / (y_max - y_min + 1e-9)
        feat_fig.add_trace(go.Scatter(
            x=filtered_df["DateTime parsed"],
            y=y,
            mode="lines",
            name=feat,
            hovertemplate=f"<b>%{{x}}</b><br>{feat}: %{{y:.3f}}<extra></extra>",
        ))

    feat_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            title="Time",
            gridcolor="rgba(255,255,255,0.05)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title="Normalized value" if normalize else "Value",
            gridcolor="rgba(255,255,255,0.05)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(feat_fig, use_container_width=True)
else:
    st.info("Select at least one feature above to plot.", icon="📈")

# -----------------------------
# DATA TABLE & DOWNLOAD
# -----------------------------
st.divider()

col_left, col_right = st.columns([3, 1])
col_left.subheader("📋 Filtered Data")
show_all_cols = col_right.toggle("Show all columns", value=False)

display_cols = (
    filtered_df.columns.tolist()
    if show_all_cols
    else ["DateTime parsed", "Predicted_RUL"]
)

PAGE_SIZE = 500
total_rows = len(filtered_df)
total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

with st.expander("Explore data table", expanded=False):
    if total_pages > 1:
        page = st.number_input(
            f"Page (1 – {total_pages},  {total_rows:,} rows total)",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        ) - 1  # convert to 0-indexed
    else:
        page = 0

    start_row = page * PAGE_SIZE
    end_row   = min(start_row + PAGE_SIZE, total_rows)

    st.caption(f"Showing rows {start_row + 1:,} – {end_row:,} of {total_rows:,}")
    st.dataframe(
        filtered_df[display_cols].iloc[start_row:end_row],
        use_container_width=True,
    )

st.download_button(
    label="⬇️ Download predictions as CSV",
    data=filtered_df[["DateTime parsed", "Predicted_RUL"]].to_csv(index=False).encode("utf-8"),
    file_name="rul_predictions.csv",
    mime="text/csv",
)