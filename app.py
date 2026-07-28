import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Cyclistic Case Study", layout="wide")

st.title("🚲 Cyclistic Case Study — Member vs Casual Riders")
st.caption("Upload your dataset (CSV or Excel) to run the analysis")

# ---------------------------------------------------------------
# File upload (replaces the old hardcoded CSV_PATH)
# ---------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload trip data file",
    type=["csv", "xlsx", "xls"],
    help="Must contain columns: started_at, ended_at, start_lat, start_lng, "
         "end_lat, end_lng, rideable_type, member_casual",
)

if uploaded_file is None:
    st.info("👆 Upload a .csv or .xlsx file to see the analysis.")
    st.stop()

REQUIRED_COLS = {
    "started_at", "ended_at", "start_lat", "start_lng",
    "end_lat", "end_lng", "rideable_type", "member_casual",
}


# @st.cache_data means this heavy load + feature engineering only runs ONCE
# per uploaded file, not on every rerun (every click/slider move reruns the
# whole script in Streamlit). Keyed on file bytes + name so a new upload
# invalidates the cache automatically.
@st.cache_data
def load_data(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])

    df["ride_length_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df["day_of_week"] = df["started_at"].dt.dayofweek  # 0 = Mon
    df["start_hour"] = df["started_at"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    df["distance_km"] = haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"])

    # drop bad rows: negative/zero/absurd durations, missing coords
    df = df[(df["ride_length_min"] > 0) & (df["ride_length_min"] < 24 * 60)]
    df = df.dropna(subset=["distance_km"])
    return df


# @st.cache_resource is for objects like trained models -- also runs ONCE
# per dataframe and is cached across reruns, so the model isn't retrained
# on every click.
@st.cache_resource
def train_model(df):
    df_model = pd.get_dummies(df, columns=["rideable_type"], prefix="bike")
    feature_cols = ["ride_length_min", "day_of_week", "start_hour", "is_weekend", "distance_km"] + \
                   [c for c in df_model.columns if c.startswith("bike_")]

    X = df_model[feature_cols]
    y = (df_model["member_casual"] == "member").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, acc, importances


# ---------------------------------------------------------------
# App layout
# ---------------------------------------------------------------
try:
    with st.spinner("Loading ride data..."):
        df = load_data(uploaded_file.getvalue(), uploaded_file.name)
except Exception as e:
    st.error(f"Couldn't process this file: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total rides", f"{len(df):,}")
col2.metric("Member rides", f"{(df['member_casual'] == 'member').sum():,}")
col3.metric("Casual rides", f"{(df['member_casual'] == 'casual').sum():,}")

st.divider()

st.subheader("Ride patterns by rider type")
tab1, tab2, tab3 = st.tabs(["Ride length", "Hour of day", "Day of week"])

with tab1:
    st.bar_chart(df.groupby("member_casual")["ride_length_min"].median())
    st.caption("Median ride length (minutes) by rider type")

with tab2:
    hourly = df.groupby(["start_hour", "member_casual"]).size().unstack(fill_value=0)
    st.line_chart(hourly)

with tab3:
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby(["day_of_week", "member_casual"]).size().unstack(fill_value=0)
    dow.index = dow_names
    st.bar_chart(dow)

st.divider()

st.subheader("Bike type usage")
bike_split = pd.crosstab(df["rideable_type"], df["member_casual"], normalize="columns")
st.bar_chart(bike_split)

st.divider()

# Map is SAMPLED, not the full dataset -- plotting every point at once
# is the other common cause of a Streamlit app hanging/freezing.
st.subheader("Ride start locations (sample)")
map_sample_size = st.slider("Number of points to show on map", 500, 10000, 3000, step=500)
map_df = (
    df[["start_lat", "start_lng"]]
    .dropna()
    .sample(min(map_sample_size, len(df)), random_state=1)
    .rename(columns={"start_lat": "lat", "start_lng": "lon"})
)
st.map(map_df)

st.divider()

st.subheader("Model: predicting member vs casual from ride features")
with st.spinner("Training model..."):
    model, acc, importances = train_model(df)

st.write(f"**Test accuracy:** {acc:.3f}")
st.bar_chart(importances)
st.caption("Feature importance — what most separates members from casual riders")
