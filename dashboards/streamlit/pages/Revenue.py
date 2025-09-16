# pages/3_Revenue.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.header("Revenue Analysis")

# ---------- dataset check ----------
if "df" not in st.session_state:
    st.error("Dataset not loaded. Run main.py first.")
    st.stop()

df = st.session_state.df.copy()

# ensure order_datetime and revenue exist
if "order_datetime" not in df.columns:
    if "order_date" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_date"], errors="coerce")
    else:
        st.error("No order datetime column found.")
        st.stop()

if "revenue" not in df.columns:
    st.error("No `revenue` column found.")
    st.stop()

# ---------- sidebar: date range ----------
st.sidebar.header("Revenue filters")
min_date = df["order_datetime"].min().date()
max_date = df["order_datetime"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select start and end date.")
    st.stop()
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df = df[(df["order_datetime"] >= start_dt) & (df["order_datetime"] <= end_dt)]

# optional segmentation: state or city
geo_col = None
for cand in ("state", "ship_state", "shipping_state", "customer_state", "city"):
    if cand in df.columns:
        geo_col = cand
        break

st.markdown("----")

# ---------- KPIs for selected period ----------
st.subheader("Top-line revenue KPIs (selected period)")
total_rev = df["revenue"].sum()
total_orders = len(df)
aov = total_rev / total_orders if total_orders else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total revenue (INR)", f"₹{total_rev:,.0f}")
col2.metric("Total orders", f"{total_orders:,}")
col3.metric("Avg order value (AOV)", f"₹{aov:,.2f}")

st.markdown("----")

# ---------- Yearly revenue + YoY ----------
st.subheader("Yearly revenue & YoY growth")
df["year"] = df["order_datetime"].dt.year
yearly = df.groupby("year")["revenue"].sum().reset_index().sort_values("year")
yearly["yoy_pct"] = yearly["revenue"].pct_change() * 100
# pretty table
yearly_display = yearly.copy()
yearly_display["revenue"] = yearly_display["revenue"].map(lambda x: f"₹{x:,.0f}")
yearly_display["yoy_pct"] = yearly_display["yoy_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
st.dataframe(yearly_display.reset_index(drop=True), width=800)

# line chart
fig_year = px.line(yearly, x="year", y="revenue", markers=True, title="Yearly Revenue")
st.plotly_chart(fig_year, use_container_width=True)

st.markdown("----")

# ---------- Monthly heatmap (years x months) ----------
st.subheader("Monthly revenue heatmap")
df["month"] = df["order_datetime"].dt.month
heat = df.groupby(["year", "month"])["revenue"].sum().reset_index()
if not heat.empty:
    heat_pivot = heat.pivot(index="year", columns="month", values="revenue").fillna(0)
    # normalize for color scaling or keep absolute
    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x="Month", y="Year", color="Revenue"),
        x=[calendar_month for calendar_month in heat_pivot.columns],
        y=heat_pivot.index,
        title="Revenue: Year vs Month",
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Not enough data to build heatmap.")

st.markdown("----")

# ---------- Top geographies ----------
if geo_col:
    st.subheader(f"Top {geo_col.title()} by revenue")
    top_n = st.slider("Top N locations", 5, 30, 10, key="top_geo_n")
    geo_rev = df.groupby(geo_col)["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(top_n)
    st.bar_chart(geo_rev.set_index(geo_col))
else:
    st.info("No geographic column detected; skip top-location analysis.")

st.markdown("----")

# ---------- download revenue summary ----------
st.subheader("Download revenue summary")
summary = df.groupby(df["order_datetime"].dt.to_period("M"))["revenue"].sum().reset_index()
summary.columns = ["period_month", "revenue"]
summary["period_month"] = summary["period_month"].astype(str)
csv = summary.to_csv(index=False).encode("utf-8")
st.download_button("Download monthly revenue CSV", data=csv, file_name="monthly_revenue.csv", mime="text/csv")

st.markdown("----")
st.info("This page focuses strictly on revenue: KPIs, yearly trends, monthly heatmap, and top locations if available.")
