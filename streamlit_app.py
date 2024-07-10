import streamlit as st


st.markdown("""
<Date Today>
""")

st.title("Today's Insight/Metrics")

# Embed the Dash app using an iframe
dash_app_url = "http://127.0.0.1:8050/"  # URL where the Dash app is running
st.components.v1.iframe(dash_app_url, width=2100, height=30000)
