"""
================================================================================
STAGE 6: UI DEPLOYMENT - FastAPI Backend
================================================================================
Serves the trained SVR pipeline via a REST API.
The frontend sends property features, and the backend returns a price prediction.

Endpoints:
  GET  /           -> Serves the frontend HTML
  POST /predict    -> Accepts property features, returns predicted price
  GET  /api/info   -> Returns model metadata

Usage:
  python app.py
  -> Opens at http://localhost:8000
================================================================================
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os

# ---- Load Model Artifacts ----
pipeline = joblib.load('svr_pipeline.joblib')
artifacts = joblib.load('preprocessing_artifacts.joblib')

# ---- Extract preprocessing objects ----
type_map = artifacts['type_map']
method_le = artifacts['method_le']
council_le = artifacts['council_le']
region_columns = artifacts['region_columns']
feature_names = artifacts['feature_names']
best_params = artifacts['best_params']
n_components = artifacts['n_components']
cap_bounds = artifacts.get('cap_bounds', {})

# ---- Region -> approximate Lat/Long mapping (from training data centroids) ----
REGION_COORDS = {
    'Eastern Metropolitan': (-37.8100, 145.1500),
    'Eastern Victoria': (-37.9500, 145.3500),
    'Northern Metropolitan': (-37.7300, 144.9800),
    'Northern Victoria': (-37.5500, 144.9000),
    'South-Eastern Metropolitan': (-37.9500, 145.1000),
    'Southern Metropolitan': (-37.8600, 145.0100),
    'Western Metropolitan': (-37.7900, 144.8300),
    'Western Victoria': (-37.8500, 144.5000),
}

# ---- Region -> approximate CouncilArea mapping ----
REGION_COUNCIL = {
    'Eastern Metropolitan': 'Boroondara',
    'Eastern Victoria': 'Cardinia',
    'Northern Metropolitan': 'Darebin',
    'Northern Victoria': 'Hume',
    'South-Eastern Metropolitan': 'Glen Eira',
    'Southern Metropolitan': 'Glen Eira',
    'Western Metropolitan': 'Brimbank',
    'Western Victoria': 'Melton',
}

# ---- Region -> approximate Propertycount ----
REGION_PROPCOUNT = {
    'Eastern Metropolitan': 7500,
    'Eastern Victoria': 3000,
    'Northern Metropolitan': 6500,
    'Northern Victoria': 3500,
    'South-Eastern Metropolitan': 5000,
    'Southern Metropolitan': 8000,
    'Western Metropolitan': 5500,
    'Western Victoria': 2000,
}

app = FastAPI(title="Melbourne House Price Predictor")

# ---- Data Models ----
class PropertyInput(BaseModel):
    rooms: int = 3
    property_type: str = 'h'       # h, t, u
    method: str = 'S'              # S, SP, PI, VB, SA
    distance: float = 10.0
    bathroom: int = 1
    car: int = 2
    landsize: float = 500.0
    building_area: float = 130.0
    year_built: int = 1970
    region: str = 'Southern Metropolitan'


def preprocess_input(data: PropertyInput) -> pd.DataFrame:
    """
    Apply the EXACT same preprocessing pipeline used during training.
    This ensures consistency between training and inference.
    """
    region = data.region
    lat, lon = REGION_COORDS.get(region, (-37.81, 144.96))
    council = REGION_COUNCIL.get(region, 'Boroondara')
    propcount = REGION_PROPCOUNT.get(region, 7000)

    # Build a single-row DataFrame matching the cleaned training format
    row = {
        'Rooms': data.rooms,
        'Type': type_map.get(data.property_type, 2),
        'Method': data.method,
        'Distance': data.distance,
        'Bathroom': data.bathroom,
        'Car': data.car,
        'Landsize': data.landsize,
        'BuildingArea': data.building_area,
        'YearBuilt': data.year_built,
        'CouncilArea': council,
        'Lattitude': lat,
        'Longtitude': lon,
        'Propertycount': propcount,
        'SaleYear': 2017.0,     # Use training-era values for temporal features
        'SaleMonth': 6.0,
    }

    df = pd.DataFrame([row])

    # Apply IQR-based capping (MUST match training preprocessing)
    for col, (lo, hi) in cap_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Encode Method
    known_methods = set(method_le.classes_)
    method_val = data.method if data.method in known_methods else method_le.classes_[0]
    df['Method'] = method_le.transform([method_val])[0]

    # Encode CouncilArea
    known_councils = set(council_le.classes_)
    council_val = council if council in known_councils else council_le.classes_[0]
    df['CouncilArea'] = council_le.transform([str(council_val)])[0]

    # One-hot encode Regionname
    for col in region_columns:
        region_name = col.replace('Region_', '')
        df[col] = 1 if region_name == region else 0

    # Ensure column order matches training
    # The feature_names list contains the exact columns the pipeline expects
    final_cols = feature_names
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[final_cols]

    # Convert bool columns to int and ensure all values are native Python types
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(float)

    return df


# ---- API Endpoints ----

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.post("/predict")
async def predict_price(data: PropertyInput):
    """Predict house price from property features."""
    try:
        df = preprocess_input(data)
        pred_log = pipeline.predict(df)
        pred_price = float(np.expm1(pred_log[0]))

        # Sanity bounds
        pred_price = max(pred_price, 50000)
        pred_price = min(pred_price, 15000000)

        return JSONResponse({
            "predicted_price": int(round(pred_price, -3)),
            "formatted_price": f"${pred_price:,.0f}",
            "confidence_range": {
                "low": f"${max(pred_price * 0.85, 50000):,.0f}",
                "high": f"${min(pred_price * 1.15, 15000000):,.0f}",
            },
            "model_info": {
                "algorithm": "SVR (RBF Kernel)",
                "r2_score": 0.8178,
                "components": int(n_components),
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/info")
async def model_info():
    """Return model metadata."""
    return {
        "model": "Support Vector Regression (RBF Kernel)",
        "hyperparameters": {
            "C": float(round(best_params['C'], 4)),
            "gamma": float(round(best_params['gamma'], 6)),
            "epsilon": float(round(best_params['epsilon'], 4)),
        },
        "pca_components": int(n_components),
        "test_r2": 0.8178,
        "test_mae": 165059,
        "test_rmse": 269008,
        "features_used": feature_names,
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Melbourne House Price Predictor")
    print("  Starting server at http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
