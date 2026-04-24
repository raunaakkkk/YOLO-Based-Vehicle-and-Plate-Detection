import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="KnightSight EdgeVision", layout="wide")

# Hide Streamlit's default headers, footers and top padding
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 0rem;
                padding-bottom: 0rem;
                padding-left: 0rem;
                padding-right: 0rem;
            }
            iframe {
                border: none;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open('knightsight.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Render the single-page HTML file inside Streamlit
components.html(html_content, height=1000, scrolling=True)
