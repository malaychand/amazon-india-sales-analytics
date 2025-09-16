# pages/6_Operations.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.header("Operations — Delivery & Returns")

# ---------- dataset ----------
if "df" not in st.session_state:
    st.error("Dataset not loaded. Run main.py first.")
    st.stop()

df = st.session_state.df.copy()

# ensure order_datetime present
if "order_datetime" not in df.columns and "order_date" in df.columns:
    df["order_datetime"] = pd.to_datetime(df["order_date"], errors="coerce")

# date filter
st.sidebar.header("Filters")
min_date = df["order_datetime"].min().date()
max_date = df["order_datetime"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select start and end date.")
    st.stop()
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df = df[(df["order_datetime"] >= start_dt) & (df["order_datetime"] <= end_dt)]

st.markdown("----")

# ---------- delivery_days cleaning ----------
if "delivery_days" in df.columns:
    def parse_delivery(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ("", "nan", "none"):
            return np.nan
        # same day / same -> 0
        if "same" in s:
            return 0.0
        # express -> treat as 0.5 day (fast) — you can change this mapping
        if "express" in s:
            return 0.5
        # negative or weird markers -> NaN
        if s.startswith("-"):
            return np.nan
        # ranges like "1-2" -> average
        nums = re.findall(r"\d+", s)
        if nums:
            nums = list(map(float, nums))
            return float(np.mean(nums))
        # fallback numeric conversion
        try:
            return float(s)
        except:
            return np.nan

    df["_delivery_days_num"] = df["delivery_days"].apply(parse_delivery)
    total = len(df)
    nonnull = df["_delivery_days_num"].notna().sum()

    st.subheader("Delivery days")
    st.write(f"Values parsed: **{nonnull:,}** / **{total:,}**")
    # summary
    st.write(df["_delivery_days_num"].describe().to_frame().T)

    # histogram
    fig = px.histogram(df, x="_delivery_days_num", nbins=30, title="Delivery days distribution")
    st.plotly_chart(fig, use_container_width=True)

    # percentages within thresholds
    within_1 = df["_delivery_days_num"].le(1).sum()
    within_2 = df["_delivery_days_num"].le(2).sum()
    within_3 = df["_delivery_days_num"].le(3).sum()
    col1, col2, col3 = st.columns(3)
    
    st.markdown("----")
    st.subheader("Slowest deliveries (sample)")
    slow = df[df["_delivery_days_num"].notna()].sort_values("_delivery_days_num", ascending=False).head(200)
    cols_show = ["order_datetime", "transaction_id", "customer_id", "delivery_days", "_delivery_days_num", "revenue"]
    cols_show = [c for c in cols_show if c in slow.columns]
    st.dataframe(slow[cols_show].head(200))
    csv = slow[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button("Download slow deliveries (CSV)", data=csv, file_name="slow_deliveries.csv", mime="text/csv")
else:
    st.info("No `delivery_days` column found.")

st.markdown("----")


# ---------- returns / return_status ----------
if "return_status" in df.columns:
    st.subheader("Return / order status")
    # show value counts (top)
    vals = df["return_status"].astype(str).value_counts().reset_index()
    vals.columns = ["status", "count"]
    st.dataframe(vals)
    csvr = vals.to_csv(index=False).encode("utf-8")
    st.download_button("Download return status counts", data=csvr, file_name="return_status_counts.csv", mime="text/csv")
else:
    st.info("No `return_status` column found.")

st.markdown("----")

