# pages/2_EDA.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Exploratory Data Analysis")

# --------- dataset check ----------
if "df" not in st.session_state:
    st.error("Dataset not loaded. Run main.py first.")
    st.stop()

# work on a copy
df = st.session_state.df.copy()

# ensure canonical datetime exists
if "order_datetime" not in df.columns:
    if "order_date" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_date"], errors="coerce")
    else:
        st.error("No order date column found.")
        st.stop()

# ensure revenue exists
if "revenue" not in df.columns:
    st.error("No `revenue` column found.")
    st.stop()

# --------- sidebar filters (date only) ----------
st.sidebar.header("Filters")
min_date = df["order_datetime"].min().date()
max_date = df["order_datetime"].max().date()
date_range = st.sidebar.date_input(
    "Order date range", [min_date, max_date], min_value=min_date, max_value=max_date
)
if len(date_range) != 2:
    st.sidebar.error("Select a valid start and end date.")
    st.stop()

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# apply date filter
df = df[(df["order_datetime"] >= start_dt) & (df["order_datetime"] <= end_dt)]

st.markdown("----")

# --------- KPIs ----------
st.subheader("Key metrics")
total_orders = len(df)
total_revenue = df["revenue"].sum()
avg_order_value = df["revenue"].mean() if total_orders else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total orders", f"{total_orders:,}")
c2.metric("Total revenue (INR)", f"₹{total_revenue:,.0f}")
c3.metric("Avg order value (AOV)", f"₹{avg_order_value:,.2f}")

st.markdown("----")

# --------- time series ----------
st.subheader("Trends over time")
agg_choice = st.selectbox("Aggregation", ["Monthly", "Quarterly", "Yearly"], index=0)

if agg_choice == "Monthly":
    rev_ts = df.set_index("order_datetime").resample("M")["revenue"].sum().reset_index()
    orders_ts = df.set_index("order_datetime").resample("M").size().reset_index(name="orders")
elif agg_choice == "Quarterly":
    rev_ts = df.set_index("order_datetime").resample("Q")["revenue"].sum().reset_index()
    orders_ts = df.set_index("order_datetime").resample("Q").size().reset_index(name="orders")
else:
    rev_ts = df.set_index("order_datetime").resample("Y")["revenue"].sum().reset_index()
    orders_ts = df.set_index("order_datetime").resample("Y").size().reset_index(name="orders")

fig_r = px.line(rev_ts, x="order_datetime", y="revenue", title=f"{agg_choice} Revenue", markers=True)
fig_o = px.line(orders_ts, x="order_datetime", y="orders", title=f"{agg_choice} Orders", markers=True)

st.plotly_chart(fig_r, use_container_width=True)
st.plotly_chart(fig_o, use_container_width=True)

st.markdown("----")

# --------- top products ----------
prod_col = None
for cand in ("product_title", "product_name", "title", "product"):
    if cand in df.columns:
        prod_col = cand
        break

if prod_col:
    st.subheader("Top products by revenue")
    top_n_prod = st.slider("Top products (N)", 5, 30, 10, key="top_prod_n")
    prod_rev = (
        df.groupby(prod_col)["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(top_n_prod)
    )
    st.dataframe(prod_rev.reset_index(drop=True))

st.markdown("----")

# --------- preview & download ----------
st.subheader("Data preview & export")
preview_cols = ["order_datetime", "transaction_id", "customer_id", "revenue"]
preview_cols = [c for c in preview_cols if c in df.columns]

st.dataframe(df[preview_cols].head(200))

csv = df[preview_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered preview (CSV)",
    data=csv,
    file_name="preview_filtered.csv",
    mime="text/csv",
)
