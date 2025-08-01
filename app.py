import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_modal import Modal

st.set_page_config(
    page_title="Earthquake Magnitude Predictor",
    layout="centered",
    page_icon="🌎"
)

# --- Custom CSS for a modern, appealing look ---
st.markdown("""
<style>
body { background-color: #f6f8fa; }
.big-title {
    font-size:3rem !important;
    font-weight:700 !important;
    text-align:center;
    color:#fff;
    margin-bottom:0.7em;
}
.stButton > button {
    font-size:1.3rem;
    background: linear-gradient(90deg, #3399ff 0%, #33cccc 100%);
    color: #fff;
    border-radius: 8px;
    padding: 0.7em 1.5em;
}
.result-box {
    background: rgba(0,0,0,0.7);
    border-radius: 12px;
    padding: 1.2em 2.5em;
    margin-top:2em;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    box-shadow:0 3px 10px #222;
}
hr { margin-top:1.5em; margin-bottom:1.5em; }
label, .stTextInput label, .stTextInput input, .stTextInput textarea {
    color: #fff !important;
    font-size: 1.3rem !important;
}
.stTextInput input {
    background: rgba(0,0,0,0.6) !important;
    color: #fff !important;
    font-size: 1.3rem !important;
}
.stCaption, .caption, .stMarkdown p {
    color: #fff !important;
    font-size: 1.1rem !important;
}
.stSubheader, .stMarkdown h2 {
    color: #fff !important;
    font-size: 2rem !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #000 !important;
    color: #fff !important;
    box-shadow: 0 0 12px #222;
    border-radius: 0 20px 20px 0;
    padding: 2em 1.5em;
}
</style>
""", unsafe_allow_html=True)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load sidebar image (replace with your actual image file)

main_bg_img = get_img_as_base64("dark.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url('data:image/webp;base64,{main_bg_img}');
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background: #484852;
color: #fff;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Title & Description ---
st.markdown('<div class="big-title">🌏 Earthquake Magnitude Predictor</div>', unsafe_allow_html=True)
st.write("""
Instantly predict the **magnitude** of an earthquake given event characteristics.
Your input is run through a state-of-the-art AI model trained on global seismic data (2000–2025).
""")
st.write("---")

# --- Sidebar: Project Explanation & Navigation ---
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        background: #000 !important;
        color: #fff !important;
        box-shadow: 0 0 12px #222;
        border-radius: 0 20px 20px 0;
        padding: 2em 1.5em;
    }
    .sidebar-title {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 1.2em;
        text-align: left;
    }
    .sidebar-desc {
        font-size: 1.1rem;
        color: #eee;
        margin-bottom: 2em;
    }
    .links-section {
        margin-top: 2em;
        padding-top: 1.5em;
        border-top: 1px solid #444;
    }
    .links-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 1em;
    }
    .link-item {
        margin-bottom: 0.8em;
        padding: 0.5em;
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        border-left: 3px solid #3399ff;
    }
    .link-item a {
        color: #66ccff !important;
        text-decoration: none;
        font-weight: 500;
    }
    .link-item a:hover {
        color: #99ddff !important;
        text-decoration: underline;
    }
    .link-desc {
        font-size: 0.9rem;
        color: #ccc;
        margin-top: 0.3em;
    }
    </style>
    <div class="sidebar-title">🌏 Earthquake Magnitude Predictor</div>
    <div class="sidebar-desc">
    This app predicts the magnitude of an earthquake using a machine learning model trained on global seismic data (2000–2025). Enter event details and instantly see the predicted magnitude, powered by Gradient Boosting. Explore how location, timing, and measurement quality affect predictions!
    </div>
    
    """,
    unsafe_allow_html=True
)

# Add the links section using Streamlit components
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Useful Links & Resources")

st.sidebar.markdown("""
<style>
.resource-card {
    background: linear-gradient(135deg, rgba(51,153,255,0.1) 0%, rgba(51,204,204,0.1) 100%);
    border: 1px solid rgba(51,153,255,0.3);
    border-radius: 10px;
    padding: 12px;
    margin: 8px 0;
    transition: all 0.3s ease;
    border-left: 4px solid #3399ff;
}
.resource-card:hover {
    background: linear-gradient(135deg, rgba(51,153,255,0.2) 0%, rgba(51,204,204,0.2) 100%);
    border-color: rgba(51,153,255,0.5);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(51,153,255,0.2);
}
.resource-title {
    font-weight: 600;
    font-size: 14px;
    color: #66ccff;
    text-decoration: none;
    margin-bottom: 4px;
    display: block;
}
.resource-title:hover {
    color: #99ddff;
    text-decoration: underline;
}
.resource-desc {
    font-size: 12px;
    color: #bbb;
    line-height: 1.3;
}
</style>

<div class="resource-card">
    <a href="https://earthquake.usgs.gov/" target="_blank" class="resource-title">🌎 USGS Earthquake Hazards Program</a>
    <div class="resource-desc">Real-time earthquake data and monitoring worldwide</div>
</div>

<div class="resource-card">
    <a href="https://www.emsc-csem.org/" target="_blank" class="resource-title">🇪🇺 European-Mediterranean Seismological Centre</a>
    <div class="resource-desc">European earthquake information and alerts</div>
</div>

<div class="resource-card">
    <a href="https://www.iris.edu/hq/" target="_blank" class="resource-title">🔬 IRIS - Seismology Research</a>
    <div class="resource-desc">Educational resources and research data</div>
</div>

<div class="resource-card">
    <a href="https://www.globalcmt.org/" target="_blank" class="resource-title">📊 Global CMT Catalog</a>
    <div class="resource-desc">Earthquake source mechanisms and data</div>
</div>

<div class="resource-card">
    <a href="https://www.ready.gov/earthquakes" target="_blank" class="resource-title">🛡️ Ready.gov - Earthquake Safety</a>
    <div class="resource-desc">Earthquake preparedness and safety tips</div>
</div>

<div class="resource-card">
    <a href="https://www.fema.gov/earthquake" target="_blank" class="resource-title">🚨 FEMA Earthquake Resources</a>
    <div class="resource-desc">Emergency preparedness and response</div>
</div>
""", unsafe_allow_html=True)


# --- Load the trained model ---
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# --- Get all features the model expects (from the preprocessor) ---
feature_names = model.named_steps['pre'].get_feature_names_out()

# --- UI: Smart feature grouping based on your top model features ---
# This list is now auto-generated from your model's expected features for robust prediction
ui_features = list(feature_names)

# --- UI: Collect raw features required by the model pipeline ---
# List of raw columns your pipeline expects (update as needed)
raw_features = [
    'latitude', 'longitude', 'depth', 'lat_lon_interaction', 
    'depth_mag_interaction', 'days_since_start', 'magError', 'magNst',
    'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 
    'magType', 'depth_type', 'target_bin'
]

# --- Feature explanations for non-experts ---
feature_explanations = {
    'lat_lon_interaction': 'Latitude × Longitude. Captures the combined location of the earthquake event.',
    'magError': 'Magnitude Error. The uncertainty in the measured magnitude.',
    'magNst': 'Number of Stations for Magnitude. How many stations contributed to the magnitude reading.',
    'rms': 'Root Mean Square of Travel Time Residuals. Indicates the fit quality of the event location.',
    'latitude': 'Latitude. North-South position of the earthquake epicenter.',
    'depth': 'Depth (km). How deep below the surface the earthquake occurred.',
    'longitude': 'Longitude. East-West position of the earthquake epicenter.',
    'days_since_start': 'Days Since Start. Number of days since the beginning of the dataset.',
    'horizontalError': 'Horizontal Error. Uncertainty in the epicenter location (km).',
    'magType': 'Magnitude Type. The method used to calculate magnitude (e.g., mb, ml, mwr).',
    'dmin': 'Minimum Distance to Station. Closest seismic station to the event (km).',
    'depthError': 'Depth Error. Uncertainty in the depth measurement (km).',
    'depth_mag_interaction': 'Depth × Magnitude. Combines depth and magnitude to capture physical effects.',
    'depth_type': 'Depth Type. Whether the event is shallow or deep.',
    'nst': 'Number of Stations. Total stations used to locate the event.',
    'gap': 'Gap. Largest azimuthal gap between stations (degrees); lower is better coverage.'
}

# --- UI: Modern Form Layout with Sliders and Dropdowns ---
modal = Modal("Enter Earthquake Event Details", key="event_form")

with st.container():
    st.markdown("""
    <style>
    .form-container {
        background: rgba(0,0,0,0.5);
        border-radius: 16px;
        padding: 2em 2em 1em 2em;
        box-shadow: 0 4px 24px #222;
        margin-bottom: 2em;
    }
    .form-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 1em;
        text-align: center;
    }
    .form-label {
        font-size: 1rem;
        color: #eee;
        margin-bottom: 0.2em;
    }
    /* Custom styling for expanders - black 80% translucent */
    .streamlit-expanderHeader {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 20px !important;
        border: none !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 10px !important;
        margin-bottom: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stExpander"] > div:first-child {
        background: rgba(0, 0, 0, 0.8) !important;
        color: #ffffff !important;
    }
    div[data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
    <div class="form-container">
    <div class="form-title">Enter Earthquake Event Details</div>
    """, unsafe_allow_html=True)
    
    user_input = {}
    
    # Location Information (Collapsible)
    with st.expander("📍 Location Information", expanded=True):
        col1, col2 = st.columns(2)
        
        user_input['latitude'] = col1.slider(
            "Latitude", min_value=-90.0, max_value=90.0, value=0.0,
            help="North-South position of the earthquake epicenter", key="latitude"
        )
        user_input['longitude'] = col2.slider(
            "Longitude", min_value=-180.0, max_value=180.0, value=0.0,
            help="East-West position of the earthquake epicenter", key="longitude"
        )
        user_input['lat_lon_interaction'] = col1.slider(
            "Lat Lon Interaction", min_value=-16200.0, max_value=16200.0, value=0.0,
            help="Latitude × Longitude interaction", key="lat_lon_interaction"
        )
        user_input['depth'] = col2.slider(
            "Depth (km)", min_value=0.0, max_value=700.0, value=10.0,
            help="How deep below the surface the earthquake occurred", key="depth"
        )
        user_input['depth_type'] = col1.selectbox(
            "Depth Type", options=['shallow', 'deep'], 
            help="Whether the event is shallow or deep", key="depth_type"
        )
    
    # Magnitude & Quality Information
    with st.expander("📊 Magnitude & Quality Information", expanded=True):
        col1, col2 = st.columns(2)
        
        user_input['magError'] = col1.slider(
            "Magnitude Error", min_value=0.0, max_value=2.0, value=0.1,
            help="Uncertainty in the measured magnitude", key="magError"
        )
        user_input['magNst'] = col2.slider(
            "Magnitude Stations", min_value=0.0, max_value=100.0, value=10.0,
            help="Number of stations that contributed to magnitude reading", key="magNst"
        )
        user_input['magType'] = col1.selectbox(
            "Magnitude Type", options=['mb', 'ml', 'mwr', 'other'],
            help="Method used to calculate magnitude", key="magType"
        )
        user_input['rms'] = col2.slider(
            "RMS", min_value=0.0, max_value=5.0, value=0.5,
            help="Root Mean Square of travel time residuals", key="rms"
        )
    
    # Station & Error Information  
    with st.expander("🔧 Station & Error Information", expanded=False):
        col1, col2 = st.columns(2)
        
        user_input['nst'] = col1.slider(
            "Number of Stations", min_value=0.0, max_value=100.0, value=20.0,
            help="Total stations used to locate the event", key="nst"
        )
        user_input['gap'] = col2.slider(
            "Gap (degrees)", min_value=0.0, max_value=360.0, value=90.0,
            help="Largest azimuthal gap between stations", key="gap"
        )
        user_input['dmin'] = col1.slider(
            "Distance to Nearest Station", min_value=0.0, max_value=100.0, value=10.0,
            help="Closest seismic station to the event (km)", key="dmin"
        )
        user_input['horizontalError'] = col2.slider(
            "Horizontal Error", min_value=0.0, max_value=100.0, value=5.0,
            help="Uncertainty in the epicenter location (km)", key="horizontalError"
        )
        user_input['depthError'] = col1.slider(
            "Depth Error", min_value=0.0, max_value=100.0, value=5.0,
            help="Uncertainty in the depth measurement (km)", key="depthError"
        )
    
    # Temporal & Advanced Features
    with st.expander("⏰ Temporal & Advanced Features", expanded=False):
        col1, col2 = st.columns(2)
        
        user_input['days_since_start'] = col1.slider(
            "Days Since Start", min_value=0.0, max_value=10000.0, value=5000.0,
            help="Number of days since the beginning of the dataset", key="days_since_start"
        )
        user_input['depth_mag_interaction'] = col2.slider(
            "Depth Mag Interaction", min_value=0.0, max_value=100.0, value=50.0,
            help="Depth × Magnitude interaction", key="depth_mag_interaction"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Predict Button ---
if st.button("Predict Magnitude 🚀"):
    # Prepare DataFrame: include ALL raw columns the model expects!
    data_row = pd.DataFrame([{col: user_input.get(col, "") for col in raw_features}])
    
    # Add target_bin feature if model expects it (temporary fix)
    # We'll set it to a default value since we can't know the true bin during prediction
    data_row['target_bin'] = 'Bin3'  # Middle bin as default
    
    # Convert numeric columns to float
    for col in data_row.columns:
        if col not in ['magType', 'depth_type', 'target_bin']:
            try:
                data_row[col] = data_row[col].astype(float)
            except Exception:
                data_row[col] = 0.0
    
    try:
        # --- Predict ---
        magnitude = model.predict(data_row)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("The model expects different features. Please check the model training.")
        magnitude = None
    if magnitude is not None:
        st.markdown(f'<div class="result-box">🌋 <b>Predicted Magnitude:</b> {magnitude:.3f}</div>', unsafe_allow_html=True)
        st.info("This prediction is powered by a Gradient Boosting model trained on global earthquake records (2000–2025). Try different values to see how magnitude changes!")
    else:
        st.warning("Unable to make prediction due to model configuration issues.")

st.caption("Created by Jarrod, 2025 · Powered by Scikit-learn · Data: USGS")

# Example side nav bar (Streamlit sidebar)


# Load earthquake data
quake_df = pd.read_csv("usgs_earthquake_data_2000_2025.csv")

# --- Sidebar filters for earthquake data ---
min_mag, max_mag = st.sidebar.slider("Min Magnitude", float(quake_df["mag"].min()), float(quake_df["mag"].max()), float(quake_df["mag"].min()), key="filter_min_mag"), st.sidebar.slider("Max Magnitude", float(quake_df["mag"].min()), float(quake_df["mag"].max()), float(quake_df["mag"].max()), key="filter_max_mag")
min_depth, max_depth = st.sidebar.slider("Min Depth", float(quake_df["depth"].min()), float(quake_df["depth"].max()), float(quake_df["depth"].min()), key="filter_min_depth"), st.sidebar.slider("Max Depth", float(quake_df["depth"].min()), float(quake_df["depth"].max()), float(quake_df["depth"].max()), key="filter_max_depth")

filtered_quake_df = quake_df[(quake_df["mag"] >= min_mag) & (quake_df["mag"] <= max_mag) & (quake_df["depth"] >= min_depth) & (quake_df["depth"] <= max_depth)]

# Only show Geographic Distribution plot (limit to 1000 rows for performance)
geo_df = filtered_quake_df.copy()
if len(geo_df) > 1000:
    geo_df = geo_df.sample(1000, random_state=42)
with st.container():
    st.header("Geographic Distribution with small dataset")
    st.plotly_chart(
        px.scatter_geo(
            geo_df,
            lat="latitude",
            lon="longitude",
            color="mag",
            hover_name="place",
            title="Earthquake Locations (2000–2025)"
        ),
        key="quake_geo"
    )
st.write("---")
st.markdown("""
### **Earthquake Feature Guide: What Each Input Means**

---

**latitude** *(Location)*  
This is how far north or south the earthquake happened, like coordinates on a map. For example, a quake at latitude 0 is right on the equator (think Indonesia), while +45 would be somewhere like northern Japan. Where the quake is matters because different parts of the world have different geology—some places are more prone to earthquakes than others.

---

**longitude** *(Location)*  
Longitude tells you how far east or west the quake was. For example, longitudes near 120 are in Asia, while -120 is near the west coast of North America. This helps pinpoint the exact location, which is important since some regions (like the Pacific “Ring of Fire”) see more earthquakes.

---

**lat_lon_interaction** *(Location)*  
This combines latitude and longitude to create a unique “location signature” for each quake. For instance, quakes in California (latitude ~36, longitude ~-120) might behave differently from those in Turkey or New Zealand, and this feature helps the model notice those regional patterns.

---

**depth** *(Depth)*  
Depth is how far underground the earthquake started, measured in kilometers. A shallow quake (like 10 km deep) will usually feel much stronger on the surface than a deep quake (like 500 km), even if the magnitude is the same. This is why some small quakes are felt widely, while others aren’t noticed at all.

---

**depth_mag_interaction** *(Depth)*  
This is simply “depth × magnitude,” which captures how the combination of being strong *and* deep (or shallow) might influence the impact. For example, a big, shallow quake is often more destructive than a big, deep one.

---

**depth_type** *(Type)*  
Classifies the earthquake as “shallow” or “deep” based on the depth. Shallow quakes (usually less than 70 km deep) are the ones that tend to cause more damage at the surface.

---

**magError** *(Magnitude)*  
This is how much uncertainty there is in the measured magnitude. If the error is low, the reported magnitude (like 6.5) is very trustworthy. If the error is high, it’s a rough estimate—like when different people guess a weight and can’t agree.

---

**magNst** *(Magnitude)*  
The number of stations that detected and measured the earthquake’s magnitude. More stations means a more reliable reading. Think of it like getting a second, third, and fourth opinion—if everyone agrees, you can trust the result!

---

**magType** *(Type)*  
This tells you which method was used to calculate the magnitude, like “mb,” “ml,” or “mwr.” Different methods are used for different sizes or distances. For example, some are better for local small quakes, others for big ones far away.

---

**days_since_start** *(Time)*  
How many days have passed since the start of the dataset (e.g., since 2000). This helps the model pick up on trends over time, such as improvements in measurement technology or changes in earthquake activity.

---

**horizontalError** *(Quality)*  
Shows how accurately the epicenter (where the quake started) was pinpointed on the surface. Smaller values mean the location is precise. Imagine trying to guess where a sound came from with your eyes closed—the more guesses you get, the smaller your error!

---

**depthError** *(Quality)*  
How accurate the measured depth is. A low value means we’re pretty sure how deep the quake was; a high value means it’s more of an estimate.

---

**rms** *(Quality)*  
This stands for “Root Mean Square” of residuals—it’s a measure of how well all the different seismic stations agreed about the location and timing of the quake. Smaller RMS means the stations all pretty much agree; a big RMS means there’s more uncertainty.

---

**gap** *(Quality)*  
This measures the largest angle between stations used to locate the earthquake. Smaller gaps are better, because it means the quake was surrounded by instruments, making the location more accurate. A big gap means there’s a “hole” in the coverage.

---

**nst** *(Quality)*  
The total number of seismic stations used to pinpoint the earthquake’s location. More is better—think of trying to locate a noise with two ears vs. ten ears!

---

**dmin** *(Quality)*  
This is the distance (in km) from the quake to the nearest seismic station. The closer a station is, the more accurate its reading—like sitting right next to a speaker vs. hearing it from the next room.

---

### **Why All This Matters**

Each feature above gives the model more context and lets it “see” the earthquake from different perspectives: where it was, how it was measured, and how reliable the measurements are. That’s how the model can make really accurate predictions—even for complex, real-world data.

""")

