# pages/4_Customers.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

st.header("Customer Analytics")

# ---------- dataset ----------
if "df" not in st.session_state:
    st.error("Dataset not loaded. Run main.py first.")
    st.stop()

df = st.session_state.df.copy()

if "order_datetime" not in df.columns:
    if "order_date" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_date"], errors="coerce")
    else:
        st.error("No order datetime found.")
        st.stop()

if "revenue" not in df.columns:
    st.error("No revenue column found.")
    st.stop()

# ---------- sidebar filters ----------
st.sidebar.header("Customer filters")
min_date = df["order_datetime"].min().date()
max_date = df["order_datetime"].max().date()
date_range = st.sidebar.date_input("Order date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select valid start and end date.")
    st.stop()

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df = df[(df["order_datetime"] >= start_dt) & (df["order_datetime"] <= end_dt)]

# identify customer id column
cust_col = None
for cand in ("customer_id", "cust_id", "buyer_id", "user_id"):
    if cand in df.columns:
        cust_col = cand
        break

if not cust_col:
    st.error("No customer id column found.")
    st.stop()

st.markdown("----")

# ---------- Top customers ----------
st.subheader("Top customers by revenue")
top_n = st.slider("Top N customers", 5, 50, 10, key="top_cust_n")
cust_rev = df.groupby(cust_col)["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(top_n)
cust_rev = cust_rev.rename(columns={cust_col: "customer_id", "revenue": "total_revenue"})
st.dataframe(cust_rev.reset_index(drop=True))
csv = cust_rev.to_csv(index=False).encode("utf-8")
st.download_button("Download top customers CSV", data=csv, file_name="top_customers.csv", mime="text/csv")

st.markdown("----")

# ---------- RFM calculation ----------
st.subheader("RFM segmentation (Recency, Frequency, Monetary)")

snapshot_date = df["order_datetime"].max() + pd.Timedelta(days=1)
rfm = df.groupby(cust_col).agg(
    recency_days = ("order_datetime", lambda x: (snapshot_date - x.max()).days),
    frequency = ("transaction_id", lambda x: x.nunique()) if "transaction_id" in df.columns else ("order_datetime", "count"),
    monetary = ("revenue", "sum")
).reset_index().rename(columns={cust_col: "customer_id"})

# small safety: convert frequency column name collision if needed
if "frequency" not in rfm.columns:
    if "order_datetime" in rfm.columns:
        rfm = rfm.rename(columns={"order_datetime": "frequency"})

# RFM scores (1-5)
rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
rfm["rfm_sum"] = rfm[["r_score","f_score","m_score"]].sum(axis=1)

# simple segments by rfm_sum
def rfm_segment(x):
    if x >= 13:
        return "Champions"
    if x >= 10:
        return "Loyal"
    if x >= 7:
        return "Promising"
    if x >= 4:
        return "At Risk"
    return "Lost"

rfm["segment"] = rfm["rfm_sum"].apply(rfm_segment)

st.write("RFM sample (top 20 by monetary):")
st.dataframe(rfm.sort_values("monetary", ascending=False).head(20).reset_index(drop=True))

csv = rfm.to_csv(index=False).encode("utf-8")
st.download_button("Download full RFM table", data=csv, file_name="rfm_table.csv", mime="text/csv")

st.markdown("----")

# ---------- RFM summary viz ----------
st.subheader("RFM segment distribution")
seg_counts = rfm["segment"].value_counts().reset_index()
seg_counts.columns = ["segment","count"]
fig_seg = px.bar(seg_counts, x="segment", y="count", title="Customer segments (RFM)")
st.plotly_chart(fig_seg, use_container_width=True)

st.markdown("----")

# ---------- Customer LTV and frequency distribution ----------
st.subheader("Customer lifetime / frequency")
ltv = rfm[["customer_id","monetary"]].copy().sort_values("monetary", ascending=False).head(20)
st.write("Top customers by lifetime value")
st.dataframe(ltv.reset_index(drop=True))

# frequency histogram
fig_freq = px.histogram(rfm, x="frequency", nbins=40, title="Customer purchase frequency distribution")
st.plotly_chart(fig_freq, use_container_width=True)

st.markdown("----")

# ---------- Cohort-ish: first purchase month and orders per cohort ----------
st.subheader("Cohort snapshot (first purchase month)")
df_cust = df.copy()
df_cust["order_month"] = df_cust["order_datetime"].dt.to_period("M")
first_purchase = df_cust.groupby(cust_col)["order_datetime"].min().reset_index().rename(columns={"order_datetime":"first_purchase"})
first_purchase["first_month"] = first_purchase["first_purchase"].dt.to_period("M").astype(str)
merged = df_cust.merge(first_purchase[[cust_col,"first_month"]], on=cust_col, how="left")
cohort = merged.groupby(["first_month", merged["order_datetime"].dt.to_period("M").astype(str)]) \
               .agg(customers=("customer_id", lambda x: x.nunique())).reset_index()
if not cohort.empty:
    # show a small sample of cohorts
    st.write(cohort.head(20))
else:
    st.info("Not enough data for cohort snapshot.")

st.markdown("----")
st.info("Customer page: top customers, RFM segmentation, LTV shortlist and a simple cohort snapshot.")
