import joblib, numpy as np, pandas as pd

pipeline = joblib.load('svr_pipeline.joblib')
artifacts = joblib.load('preprocessing_artifacts.joblib')

type_map = artifacts['type_map']
method_le = artifacts['method_le']
council_le = artifacts['council_le']
region_columns = artifacts['region_columns']
feature_names = artifacts['feature_names']
cap_bounds = artifacts.get('cap_bounds', {})

def predict(rooms, baths, car, ptype, land, building, year, dist, region):
    coords = {'Southern Metropolitan':(-37.86,145.01),'Eastern Metropolitan':(-37.81,145.15),'South-Eastern Metropolitan':(-37.95,145.10)}
    councils = {'Southern Metropolitan':'Glen Eira','Eastern Metropolitan':'Boroondara','South-Eastern Metropolitan':'Glen Eira'}
    propcounts = {'Southern Metropolitan':8000,'Eastern Metropolitan':7500,'South-Eastern Metropolitan':5000}
    lat,lon = coords.get(region,(-37.81,144.96))
    council = councils.get(region,'Boroondara')
    propcount = propcounts.get(region,7000)
    row = {'Rooms':rooms,'Type':type_map[ptype],'Method':0,'Distance':dist,'Bathroom':baths,
           'Car':car,'Landsize':land,'BuildingArea':building,'YearBuilt':year,
           'CouncilArea':0,'Lattitude':lat,'Longtitude':lon,'Propertycount':propcount,
           'SaleYear':2017.0,'SaleMonth':6.0}
    df = pd.DataFrame([row])
    for col,(lo,hi) in cap_bounds.items():
        if col in df.columns: df[col] = df[col].clip(lower=lo,upper=hi)
    df['Method'] = method_le.transform(['S'])[0]
    known = set(council_le.classes_)
    cv = council if council in known else council_le.classes_[0]
    df['CouncilArea'] = council_le.transform([str(cv)])[0]
    for col in region_columns:
        rn = col.replace('Region_','')
        df[col] = 1 if rn == region else 0
    for col in feature_names:
        if col not in df.columns: df[col] = 0
    df = df[feature_names]
    for col in df.columns:
        if df[col].dtype == bool: df[col] = df[col].astype(int)
        elif df[col].dtype in ['int64','int32']: df[col] = df[col].astype(float)
    pred = np.expm1(pipeline.predict(df))[0]
    return pred

tests = [
    ('Modest',     2, 1, 1, 'h', 300, 90,  1960, 15, 'Southern Metropolitan'),
    ('Average',    3, 1, 2, 'h', 500, 130, 1970, 10, 'Southern Metropolitan'),
    ('Upscale',    4, 2, 2, 'h', 700, 200, 2000, 7,  'Southern Metropolitan'),
    ('Premium',    5, 3, 3, 'h', 900, 300, 2010, 5,  'Southern Metropolitan'),
    ('Luxury',     6, 4, 4, 'h', 1000, 400, 2015, 3, 'Southern Metropolitan'),
    ('Extreme',   12, 8,10, 'h', 1000, 500, 2013, 3, 'South-Eastern Metropolitan'),
]
print(f"{'Label':<12} {'Rooms':>5} {'Bath':>5} {'Build':>6} {'Dist':>5} {'Predicted':>15}")
print('-' * 55)
for label, r, b, c, t, l, ba, y, d, reg in tests:
    p = predict(r, b, c, t, l, ba, y, d, reg)
    print(f"{label:<12} {r:>5} {b:>5} {ba:>6} {d:>5} ${p:>14,.0f}")
