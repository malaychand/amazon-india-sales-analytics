# pages/5_Products.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Product Analytics")

# ---------- dataset availability ----------
if "df" not in st.session_state:
    st.error("Dataset not loaded. Run main.py first.")
    st.stop()

df = st.session_state.df.copy()

# ensure essentials
if "order_datetime" not in df.columns:
    if "order_date" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_date"], errors="coerce")
    else:
        st.error("No order date/datetime column found.")
        st.stop()

if "revenue" not in df.columns:
    st.error("No `revenue` column found.")
    st.stop()

# ---------- product column detection ----------
prod_col = None
for cand in ("product_title", "product_name", "title", "product"):
    if cand in df.columns:
        prod_col = cand
        break

if not prod_col:
    st.error("No product/title column found in dataset.")
    st.stop()

# ---------- sidebar filters ----------
st.sidebar.header("Product filters")
min_date = df["order_datetime"].min().date()
max_date = df["order_datetime"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select start and end date.")
    st.stop()
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df = df[(df["order_datetime"] >= start_dt) & (df["order_datetime"] <= end_dt)]

# optional quantity / price columns
qty_col = None
for cand in ("quantity", "qty", "order_quantity"):
    if cand in df.columns:
        qty_col = cand
        break

price_col = None
for cand in ("final_amount_inr", "subtotal_inr", "discounted_price_inr", "original_price_inr", "price"):
    if cand in df.columns:
        price_col = cand
        break

# ---------- Top products ----------
st.subheader("Top products")
top_n = st.slider("Top N products", 5, 50, 15, key="top_prod_n")
prod_agg = (
    df.groupby(prod_col)
    .agg(orders=("transaction_id", "nunique") if "transaction_id" in df.columns else ("order_datetime", "count"),
         revenue=("revenue", "sum"),
         avg_price=(price_col, "mean") if price_col else ("revenue", lambda x: x.mean()))
    .reset_index()
)
prod_agg = prod_agg.sort_values("revenue", ascending=False).head(top_n).reset_index(drop=True)

st.dataframe(prod_agg)

# download
csv = prod_agg.to_csv(index=False).encode("utf-8")
st.download_button("Download top products CSV", data=csv, file_name="top_products.csv", mime="text/csv")

st.markdown("---")

# ---------- Product detail / time series ----------
st.subheader("Product detail / time series")
product_list = df[prod_col].dropna().astype(str).unique().tolist()
# keep the dropdown manageable
product_list_short = product_list[:10000]  # safety
selected = st.selectbox("Select product", options=product_list_short)

if selected:
    sel_df = df[df[prod_col].astype(str) == str(selected)].copy()
    total_rev = sel_df["revenue"].sum()
    total_orders = sel_df["transaction_id"].nunique() if "transaction_id" in sel_df.columns else len(sel_df)
    st.markdown(f"**Selected product:** {selected}")
    st.write(f"Total revenue: ₹{total_rev:,.0f}  |  Orders: {total_orders:,}")

    # time series for this product
    freq = st.selectbox("Time aggregation", ["M", "Q", "Y"], index=0, key="prod_ts_freq")
    rev_ts = sel_df.set_index("order_datetime").resample(freq)["revenue"].sum().reset_index()
    fig = px.line(rev_ts, x="order_datetime", y="revenue", title=f"Revenue for: {selected}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # quantity and price insights if available
    if qty_col:
        qty_sum = sel_df[qty_col].sum()
        st.write(f"Total quantity sold: {qty_sum:,}")
    if price_col:
        # clean price column and show distribution
        try:
            sel_df["_price_num"] = pd.to_numeric(sel_df[price_col].astype(str).str.replace(",", "").str.replace("₹", ""), errors="coerce")
            st.write("Price distribution (sample):")
            st.write(sel_df["_price_num"].describe().to_frame().T)
            fig_p = px.histogram(sel_df, x="_price_num", nbins=40, title="Price distribution")
            st.plotly_chart(fig_p, use_container_width=True)
        except Exception:
            st.info("Could not parse price values for this product.")

st.markdown("---")

# ---------- Price vs revenue scatter (top products) ----------
if price_col:
    st.subheader("Price vs Revenue (top products)")
    # build a table of avg price & revenue per product (top 200 by revenue)
    price_agg = (
        df.groupby(prod_col)
        .agg(avg_price=(price_col, lambda x: pd.to_numeric(x.astype(str).str.replace(",", "").str.replace("₹", ""), errors="coerce").mean()),
             revenue=("revenue", "sum"),
             orders=("order_datetime", "count"))
        .reset_index()
        .dropna(subset=["avg_price"])
        .sort_values("revenue", ascending=False)
        .head(200)
    )
    if not price_agg.empty:
        fig_scatter = px.scatter(price_agg, x="avg_price", y="revenue", size="orders", hover_data=[prod_col], title="Avg Price vs Revenue (top products)")
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

