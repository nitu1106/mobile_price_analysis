# TASK 2: Interactive Mobile Price Visualizations
# Run with: streamlit run task2_interactive_visualization.py

import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Mobile Price Analysis", layout="wide")
st.title("Mobile Price Analysis - Interactive Dashboard")
st.write("Explore mobile features and their relationship with price range.")

# Load dataset
def load_data():
    for filename in ["mobile_price.csv", "train.csv"]:
        if os.path.exists(filename):
            return pd.read_csv(filename)
    raise FileNotFoundError(
        "Place mobile_price.csv or train.csv in the same folder as this file."
    )

try:
    df = load_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "price_range" not in df.columns:
    st.error("Target column 'price_range' was not found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")
price_values = sorted(df["price_range"].dropna().unique().tolist())
selected_prices = st.sidebar.multiselect(
    "Select price range", price_values, default=price_values
)

filtered = df[df["price_range"].isin(selected_prices)].copy()

numeric_columns = filtered.select_dtypes(include="number").columns.tolist()
feature_options = [c for c in numeric_columns if c != "price_range"]

if not feature_options:
    st.warning("No numeric feature columns were found.")
    st.stop()

default_x = "ram" if "ram" in feature_options else feature_options[0]
default_y = "battery_power" if "battery_power" in feature_options else feature_options[0]

st.sidebar.subheader("Choose Features")
x_feature = st.sidebar.selectbox("X-axis feature", feature_options,
                                 index=feature_options.index(default_x))
y_feature = st.sidebar.selectbox("Y-axis feature", feature_options,
                                 index=feature_options.index(default_y))

# Slider filter for selected feature
min_value = float(filtered[x_feature].min())
max_value = float(filtered[x_feature].max())
if min_value < max_value:
    selected_range = st.sidebar.slider(
        f"Filter {x_feature}",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value)
    )
    filtered = filtered[
        filtered[x_feature].between(selected_range[0], selected_range[1])
    ]

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Mobiles displayed", len(filtered))
col2.metric("Features", len(feature_options))
col3.metric("Price categories", filtered["price_range"].nunique())

st.dataframe(filtered.head(100), use_container_width=True)

# Interactive scatter plot with hover tooltips
st.subheader("1. Interactive Feature Relationship")
fig_scatter = px.scatter(
    filtered, x=x_feature, y=y_feature, color="price_range",
    hover_data=feature_options,
    title=f"{x_feature} vs {y_feature}",
    labels={"price_range": "Price Range"}
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Box plot
st.subheader("2. Feature Distribution by Price Range")
box_feature = st.selectbox("Choose feature for box plot", feature_options,
                           index=feature_options.index(default_x))
fig_box = px.box(
    filtered, x="price_range", y=box_feature, color="price_range",
    points="outliers", title=f"{box_feature} by Price Range"
)
st.plotly_chart(fig_box, use_container_width=True)

# Average feature comparison
st.subheader("3. Average Feature Values")
avg_df = filtered.groupby("price_range")[feature_options].mean().reset_index()
fig_bar = px.bar(
    avg_df, x="price_range", y=feature_options,
    barmode="group", title="Average Feature Values for Each Price Range"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Tip: Hover over chart points to view details. Use the sidebar filters to explore the data.")
