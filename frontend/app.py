import streamlit as st
import requests
import io
from PIL import Image
import base64
import pandas as pd
import json

# Backend URL for models list
models_api_url = "http://localhost:8000/models/list"

@st.cache_data(ttl=60)  # cache for 60 seconds
def fetch_models(model_type: str = "helmet"):
    """
    Fetch available model names for the given type (helmet or license_plate).
    """
    try:
        # Add query param for type
        params = {"type": model_type}
        response = requests.get(models_api_url, params=params)
        response.raise_for_status()
        data = response.json()

        # If the API returned an error dict instead of list
        if isinstance(data, dict) and "error" in data:
            st.warning(f"API Error: {data['error']}")
            return ["best"]

        return data  # list of model names

    except Exception as e:
        st.error(f"Error fetching {model_type} models: {e}")
        return ["best"]  # fallback

# Configuration
API_URL = "http://localhost:8000"
DETECT_ENDPOINT = f"{API_URL}/detect"

st.set_page_config(
    page_title="Helmet Violation Detector", 
    page_icon="🏍️", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .violation-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .safe-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏍️ Helmet Violation Detection System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing motorcycles/bikes"
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Helmet Violation Model",
        fetch_models('helmet'),
        help="Choose the helmet violation detection model"
    )
    
    # Model selection
    license_model_choice = st.selectbox(
        "License Plate Model",
        fetch_models('license_plate'),
        help="Choose the license plate detection model"
    )
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Detect button
    detect_clicked = st.button("🔍 Detect Violations", type="primary")

# Main content area
if uploaded_file is None:
    st.info("👆 Please upload an image to get started")
    st.markdown("""
    ### How to use:
    1. Upload an image containing motorcycles or bikes
    2. Select your preferred detection model
    3. Adjust the confidence threshold if needed
    4. Click 'Detect Violations' to analyze the image
    
    ### What this system detects:
    - ✅ Riders wearing helmets (marked in green)
    - ❌ Riders without helmets (marked in red)
    - 🔢 License plates for violations (when visible)
    """)
else:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if detect_clicked:
        with st.spinner("🔄 Analyzing image for helmet violations..."):
            try:
                # Prepare request
                files = {"file": uploaded_file.getvalue()}
                params = {
                    "confidence": confidence,
                    "helmet_model": model_choice,
                    "license_model": license_model_choice
                }
                
                # Make API request
                response = requests.post(DETECT_ENDPOINT, files=files, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display annotated image
                    with col2:
                        st.subheader("📋 Detection Results")
                        if 'annotated_image' in result:
                            annotated_bytes = base64.b64decode(result['annotated_image'])
                            annotated_image = Image.open(io.BytesIO(annotated_bytes))
                            st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                    
                    # Display statistics
                    st.subheader("📊 Detection Summary")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Total Detections", result.get('total_detections', 0))
                    with col_stat2:
                        violations = result.get('violations', 0)
                        st.metric("Violations Found", violations)
                    with col_stat3:
                        safe_riders = result.get('total_detections', 0) - violations
                        st.metric("Safe Riders", safe_riders)
                    
                    # Display detailed results
                    detections = result.get('detections', [])
                    if detections:
                        st.subheader("🔍 Detailed Detection Results")
                        
                        for i, detection in enumerate(detections):
                            helmet_status = detection.get("helmet", "Unknown")
                            helmet_conf = detection.get("helmet_confidence", 0)
                            plate_text = detection.get("number_plate_text", "")
                            plate_conf = detection.get("number_plate_confidence", 0)
                            bike_box = detection.get("motorbike_box", [])

                            if helmet_status.lower() == "no helmet":
                                # 🚨 Violation card
                                st.markdown(f"""
                                <div class="violation-card" style="
                                    background-color:#ffe6e6;
                                    border-left:6px solid #ff4d4d;
                                    padding:10px;
                                    margin-bottom:10px;
                                    border-radius:8px;
                                ">
                                    <h4>⚠️ Violation #{i+1}</h4>
                                    <p><strong>Status:</strong> No Helmet Detected</p>
                                    <p><strong>Confidence:</strong> {helmet_conf:.2%}</p>
                                    <p><strong>License Plate:</strong> {plate_text if plate_text else 'Not detected'}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # License plate details (if available)
                                if plate_text:
                                    with st.expander(f"🔢 License Plate Details - Violation #{i+1}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**Plate Number:** {plate_text}")
                                            st.write(f"**Confidence:** {plate_conf:.2%}")
                                        with col2:
                                            st.write(f"**Bike Bounding Box:** {bike_box}")

                            else:
                                # ✅ Safe rider card
                                st.markdown(f"""
                                <div class="safe-card" style="
                                    background-color:#e6ffe6;
                                    border-left:6px solid #33cc33;
                                    padding:10px;
                                    margin-bottom:10px;
                                    border-radius:8px;
                                ">
                                    <h4>✅ Safe Rider #{i+1}</h4>
                                    <p><strong>Status:</strong> Helmet Detected</p>
                                    <p><strong>Confidence:</strong> {helmet_conf:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Export options
                        st.subheader("📥 Export Results")
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            # Create CSV data
                            csv_data = []
                            for i, det in enumerate(detections):
                                csv_data.append({
                                    'Rider_ID': i+1,
                                    'Status': det['helmet'],
                                    'Confidence': f"{det['helmet_confidence']:.2%}",
                                    'License_Plate': det.get('number_plate_text', 'N/A'),
                                    'License_Plate_Confidence': f"{det['number_plate_confidence']:.2%}",
                                    'Violation': 'Yes' if det['helmet'] == 'No Helmet' else 'No'
                                })
                            
                            df = pd.DataFrame(csv_data)
                            csv = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📊 Download CSV Report",
                                data=csv,
                                file_name="helmet_violation_report.csv",
                                mime="text/csv"
                            )
                        
                        with col_exp2:
                            # Download JSON results
                            json_str = json.dumps(result, indent=2)
                            st.download_button(
                                label="📋 Download JSON Data",
                                data=json_str,
                                file_name="detection_results.json",
                                mime="application/json"
                            )
                    
                    else:
                        st.info("No detections found in the image.")
                
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend API. Please ensure the FastAPI server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Helmet Violation Detection System | Built with Streamlit & FastAPI</p>
    <p>⚠️ This system is for demonstration purposes. Always ensure proper safety equipment while riding.</p>
</div>
""", unsafe_allow_html=True)
