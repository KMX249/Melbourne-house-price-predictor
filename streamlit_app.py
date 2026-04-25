import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- Page Configuration ----
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---- Load Models & Artifacts (Cached for speed) ----
@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load('svr_pipeline.joblib')
        artifacts = joblib.load('preprocessing_artifacts.joblib')
        return pipeline, artifacts
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'svr_pipeline.joblib' and 'preprocessing_artifacts.joblib' are in the directory.")
        st.stop()

pipeline, artifacts = load_models()

# Extract preprocessing variables
type_map = artifacts['type_map']
method_le = artifacts['method_le']
council_le = artifacts['council_le']
region_columns = artifacts['region_columns']
feature_names = artifacts['feature_names']
cap_bounds = artifacts.get('cap_bounds', {})

# ---- Geographic Lookup Tables ----
REGION_COORDS = {
    'Southern Metropolitan': (-37.86, 145.01),
    'Northern Metropolitan': (-37.73, 144.98),
    'Western Metropolitan': (-37.78, 144.85),
    'Eastern Metropolitan': (-37.81, 145.15),
    'South-Eastern Metropolitan': (-37.95, 145.10),
    'Eastern Victoria': (-37.90, 145.30),
    'Northern Victoria': (-37.50, 144.90),
    'Western Victoria': (-37.70, 144.50),
}
REGION_COUNCIL = {
    'Southern Metropolitan': 'Glen Eira',
    'Northern Metropolitan': 'Darebin',
    'Western Metropolitan': 'Brimbank',
    'Eastern Metropolitan': 'Boroondara',
    'South-Eastern Metropolitan': 'Glen Eira',
    'Eastern Victoria': 'Maroondah',
    'Northern Victoria': 'Macedon Ranges',
    'Western Victoria': 'Moorabool',
}

# ---- UI Header ----
st.title("🏠 Melbourne House Price Predictor")
st.markdown("""
Powered by **Support Vector Regression (SVR)** with PCA dimensionality reduction.
Enter the property details below to get an estimated market value.
""")

st.divider()

# ---- UI Input Form ----
with st.form("prediction_form"):
    st.subheader("Property Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rooms = st.number_input("Rooms", min_value=1, max_value=12, value=3)
        property_type = st.selectbox("Property Type", options=['House', 'Townhouse', 'Unit / Apartment'])
    
    with col2:
        bathroom = st.number_input("Bathrooms", min_value=0, max_value=8, value=1)
        method_options = ['Sold at Auction', 'Sold Prior to Auction', 'Passed In', 'Vendor Bid', 'Sold After Auction']
        method_display = st.selectbox("Sale Method", options=method_options)
        
    with col3:
        car = st.number_input("Car Spaces", min_value=0, max_value=10, value=2)

    st.subheader("Size & Age")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        landsize = st.number_input("Land Size (sqm)", min_value=0, max_value=5000, value=500, step=10)
    with col5:
        building_area = st.number_input("Building Area (sqm)", min_value=10, max_value=500, value=130, step=5)
    with col6:
        year_built = st.number_input("Year Built", min_value=1830, max_value=2025, value=1970)

    st.subheader("Location")
    col7, col8 = st.columns(2)
    
    with col7:
        distance = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    with col8:
        region = st.selectbox("Region", options=list(REGION_COORDS.keys()))

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# ---- Prediction Logic ----
if submitted:
    # 1. Map UI selections to model format
    type_code = 'h'
    if property_type == 'Townhouse': type_code = 't'
    elif property_type == 'Unit / Apartment': type_code = 'u'
        
    method_map = {'Sold at Auction': 'S', 'Sold Prior to Auction': 'SP', 'Passed In': 'PI', 
                  'Vendor Bid': 'VB', 'Sold After Auction': 'SA'}
    method_code = method_map[method_display]
    
    lat, lon = REGION_COORDS.get(region, (-37.81, 144.96))
    council = REGION_COUNCIL.get(region, 'Boroondara')

    # 2. Build the exact row dictionary expected by the pipeline
    row = {
        'Rooms': rooms,
        'Type': type_map.get(type_code, 2),
        'Method': method_code,
        'Distance': distance,
        'Bathroom': bathroom,
        'Car': car,
        'Landsize': landsize,
        'BuildingArea': building_area,
        'YearBuilt': year_built,
        'CouncilArea': council,
        'Lattitude': lat,
        'Longtitude': lon,
        'Propertycount': 7000, # Static median
        'SaleYear': 2017.0,
        'SaleMonth': 6.0,
    }
    
    df = pd.DataFrame([row])

    # 3. Apply IQR-based capping (The crucial Kernel Collapse fix)
    for col, (lo, hi) in cap_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 4. Handle Categorical Encodings
    known_methods = set(method_le.classes_)
    df['Method'] = method_le.transform([method_code if method_code in known_methods else method_le.classes_[0]])[0]
    
    known_councils = set(council_le.classes_)
    df['CouncilArea'] = council_le.transform([str(council) if council in known_councils else council_le.classes_[0]])[0]

    # 5. Handle Regionname One-Hot Encoding
    for col in region_columns:
        region_name = col.replace('Region_', '')
        df[col] = 1 if region_name == region else 0

    # 6. Ensure exact column order and types
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_names]
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(float)

    # 7. Predict!
    with st.spinner("Calculating via SVR pipeline..."):
        log_pred = pipeline.predict(df)[0]
        actual_price = np.expm1(log_pred)
        
        # Calculate dynamic confidence range (using the ~14.67% MAPE)
        margin = actual_price * 0.1467
        low_bound = actual_price - margin
        high_bound = actual_price + margin

    # 8. Display Results
    st.divider()
    st.success("Prediction Complete!")
    
    st.metric(
        label="Estimated Market Value", 
        value=f"${actual_price:,.0f}"
    )
    
    st.caption(f"Estimated Confidence Range (±14.67% MAPE): **${low_bound:,.0f} — ${high_bound:,.0f}**")
    
    # Model info boxes
    info1, info2, info3 = st.columns(3)
    info1.info("⚙️ Model: SVR (RBF Kernel)")
    info2.info("📊 Test R²: 82.0%")
    info3.info("📉 PCA Components: 18")
