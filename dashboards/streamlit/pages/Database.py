# pages/7_Database.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import sqlite3
import io

st.header("Database (SQLite) — export & quick queries")

# require cleaned df
if "df" not in st.session_state:
    st.error("Cleaned dataset not loaded. Run main.py first.")
    st.stop()

df = st.session_state.df.copy()

# DB path
DB_PATH = Path("data/amazon_analytics.db")
DB_URI = f"sqlite:///{DB_PATH.as_posix()}"

st.write("Local SQLite DB path:", str(DB_PATH))

# ---------- Write to SQLite ----------
if st.button("Save cleaned dataset to SQLite (replace table `transactions`)"):
    try:
        engine = create_engine(DB_URI, connect_args={"check_same_thread": False})
        # write in chunks to avoid memory spikes
        df.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=20000)
        st.success(f"Wrote {len(df):,} rows → table `transactions` in {DB_PATH}")
    except Exception as e:
        st.error(f"Error writing to SQLite: {e}")

st.markdown("---")

# ---------- Show table size / preview ----------
if DB_PATH.exists():
    try:
        conn = sqlite3.connect(DB_PATH.as_posix())
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        st.write("Tables in DB:", tables)
        if "transactions" in tables:
            cur.execute("SELECT COUNT(*) FROM transactions")
            cnt = cur.fetchone()[0]
            st.write(f"transactions: {cnt:,} rows")
            if st.button("Show sample 200 rows from `transactions`"):
                sample = pd.read_sql_query("SELECT * FROM transactions LIMIT 200", conn)
                st.dataframe(sample)
        conn.close()
    except Exception as e:
        st.error(f"Could not read DB: {e}")
else:
    st.info("Database file not found. Click the button above to create it.")

st.markdown("---")


