import streamlit as st

st. set_page_config(layout="wide")

# Sidebar elements
st.sidebar.markdown("### Menu")
st.sidebar.markdown("[Dashboard](#)")
st.sidebar.markdown("[AI Chat](#)")
st.sidebar.markdown("### Custom")
st.sidebar.markdown("[Extra Pages](#)")
st.sidebar.markdown("[Auth Pages](#)")
st.sidebar.markdown("[Error Pages](#)")
st.sidebar.markdown("### Elements")
st.sidebar.markdown("[Components](#)")
st.sidebar.markdown("[Forms](#)")
st.sidebar.markdown("[Maps](#)")
st.sidebar.markdown("[Tables](#)")
st.sidebar.markdown("[Chart](#)")
st.sidebar.markdown("[Level](#)")
st.sidebar.markdown("[Badge Items](#)")

# Main content
st.markdown("<Date Today>")
st.title("Today's Insight/Metrics")

# Embed the Dash app using an iframe
dash_app_url = "http://127.0.0.1:8050/"  # URL where the Dash app is running
st.components.v1.iframe(dash_app_url, width=2100, height=30000)
