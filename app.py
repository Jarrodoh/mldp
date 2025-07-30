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
Ps. Please use dark theme for the streamlit for best experience
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
    </style>
    <div class="sidebar-title">🌏 Earthquake Magnitude Predictor</div>
    <div class="sidebar-desc">
    This app predicts the magnitude of an earthquake using a machine learning model trained on global seismic data (2000–2025). Enter event details and instantly see the predicted magnitude, powered by Gradient Boosting. Explore how location, timing, and measurement quality affect predictions!
    </div>
    """,
    unsafe_allow_html=True
)


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
    'lat_lon_interaction', 'magError', 'magNst', 'rms', 'latitude', 'depth', 'longitude',
    'days_since_start', 'horizontalError', 'magType', 'dmin', 'depthError',
    'depth_mag_interaction', 'depth_type', 'nst', 'gap'
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
    </style>
    <div class="form-container">
    <div class="form-title">Enter Earthquake Event Details</div>
    """, unsafe_allow_html=True)
    user_input = {}
    slider_features = [
        'lat_lon_interaction', 'magError', 'magNst', 'rms', 'latitude', 'depth', 'longitude',
        'days_since_start', 'horizontalError', 'dmin', 'depthError', 'depth_mag_interaction', 'nst', 'gap'
    ]
    dropdown_features = {
        'magType': ['mb', 'ml', 'mwr', 'other'],
        'depth_type': ['shallow', 'deep', 'unknown']
    }
    # Split features for columns
    col1, col2 = st.columns(2)
    for i, feat in enumerate(slider_features):
        explanation = feature_explanations.get(feat, '')
        min_val, max_val = 0.0, 100.0
        if feat == 'latitude': min_val, max_val = -90.0, 90.0
        if feat == 'longitude': min_val, max_val = -180.0, 180.0
        if feat == 'depth': min_val, max_val = 0.0, 700.0
        if feat == 'magError': min_val, max_val = 0.0, 2.0
        if feat == 'rms': min_val, max_val = 0.0, 5.0
        if feat == 'gap': min_val, max_val = 0.0, 360.0
        target_col = col1 if i < len(slider_features)//2 else col2
        user_input[feat] = target_col.slider(
            label=f"{feat.replace('_', ' ').title()}",
            min_value=min_val,
            max_value=max_val,
            value=min_val,
            help=explanation,
            key=feat
        )
    # Dropdowns in second column
    for i, (feat, options) in enumerate(dropdown_features.items()):
        explanation = feature_explanations.get(feat, '')
        col = col2 if i == 0 else col1
        user_input[feat] = col.selectbox(
            label=f"{feat.replace('_', ' ').title()}",
            options=options,
            help=explanation,
            key=feat
        )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Predict Button ---
if st.button("Predict Magnitude 🚀"):
    # Prepare DataFrame: include ALL raw columns the model expects!
    data_row = pd.DataFrame([{col: user_input.get(col, "") for col in raw_features}])
    # Convert numeric columns to float
    for col in data_row.columns:
        if col not in ['magType', 'depth_type']:
            try:
                data_row[col] = data_row[col].astype(float)
            except Exception:
                data_row[col] = 0.0
    # --- Predict ---
    magnitude = model.predict(data_row)[0]
    st.markdown(f'<div class="result-box">🌋 <b>Predicted Magnitude:</b> {magnitude:.3f}</div>', unsafe_allow_html=True)
    st.info("This prediction is powered by a Gradient Boosting model trained on global earthquake records (2000–2025). Try different values to see how magnitude changes!")

st.caption("Created by Jarrod, 2025 · Powered by Scikit-learn · Data: USGS")

# Example side nav bar (Streamlit sidebar)


# Load earthquake data
quake_df = pd.read_csv("usgs_earthquake_data_2000_2025(Small sample size) Real one with submission folder.csv")

# --- Sidebar filters for earthquake data ---
min_mag, max_mag = st.sidebar.slider("Min Magnitude", float(quake_df["mag"].min()), float(quake_df["mag"].max()), float(quake_df["mag"].min()), key="filter_min_mag"), st.sidebar.slider("Max Magnitude", float(quake_df["mag"].min()), float(quake_df["mag"].max()), float(quake_df["mag"].max()), key="filter_max_mag")
min_depth, max_depth = st.sidebar.slider("Min Depth", float(quake_df["depth"].min()), float(quake_df["depth"].max()), float(quake_df["depth"].min()), key="filter_min_depth"), st.sidebar.slider("Max Depth", float(quake_df["depth"].min()), float(quake_df["depth"].max()), float(quake_df["depth"].max()), key="filter_max_depth")

filtered_quake_df = quake_df[(quake_df["mag"] >= min_mag) & (quake_df["mag"] <= max_mag) & (quake_df["depth"] >= min_depth) & (quake_df["depth"] <= max_depth)]

# Only show Geographic Distribution plot (limit to 1000 rows for performance)
geo_df = filtered_quake_df.copy()
if len(geo_df) > 1000:
    geo_df = geo_df.sample(50000, random_state=42)
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

