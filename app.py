import streamlit as st

st.set_page_config(page_title="Legal AI Pro", layout="wide")

# Simple Navigation
pg = st.navigation([
    st.Page("pages/1_RAG_Analyst.py", title="Document Analyst", icon="⚖️"),
    st.Page("pages/2_Legal_QA.py", title="Legal Advice Bot", icon="💬")
])
pg.run()