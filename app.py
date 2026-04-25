import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import cv2
import numpy as np

# Optimize CPU threading for ONNX/RapidOCR
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Page Config
st.set_page_config(page_title="KnightSight EdgeVision", layout="wide")

# Hide Streamlit's default headers, footers and style the uploader
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
            [data-testid="stFileUploader"] {
                background-color: #0A1520;
                border-bottom: 2px solid #1A3040;
                padding: 10px 20px;
                margin-bottom: 0;
            }
            [data-testid="stFileUploader"] > div > div > div > div {
                color: #00F5FF;
                font-family: 'Orbitron', sans-serif;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Cache ALPR Model Initialization
@st.cache_resource
def load_alpr_pipeline_v4():
    from inference import ALPRPipeline
    # ultralytics will auto-download yolo11n.pt if it's missing locally
    return ALPRPipeline(vehicle_weights="yolo11n.pt", plate_weights="best.pt")

pipeline = load_alpr_pipeline_v4()

# File Uploader
uploaded_file = st.file_uploader("UPLOAD VEHICLE IMAGE FOR ALPR SCAN", type=["jpg", "jpeg", "png"])

result_json = None
base64_image = ""

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process image
    print(">>> RUNNING PIPELINE V2 ON UPLOADED IMAGE <<<")
    with st.spinner("ANALYZING..."):
        result_json = pipeline.process_image(img)
    
    # Encode to base64 for injecting into HTML
    _, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

# Read HTML Content
with open('knightsight_v2.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Inject Data
if result_json is not None:
    # Replace placeholder comments with actual data
    html_content = html_content.replace(
        'let mockData = null; // INJECT_MOCK_DATA_HERE',
        f'let mockData = {json.dumps(result_json)};'
    )
    html_content = html_content.replace(
        'let uploadedImageBase64 = ""; // INJECT_IMAGE_BASE64_HERE',
        f'let uploadedImageBase64 = "{base64_image}";'
    )

# Render the HTML file inside Streamlit
components.html(html_content, height=1000, scrolling=True)
