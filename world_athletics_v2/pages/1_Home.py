"""
Home page - redirects to app.py main content.
This file exists so Streamlit shows "Home" in the sidebar navigation.
"""
import streamlit as st
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.switch_page("app.py")
