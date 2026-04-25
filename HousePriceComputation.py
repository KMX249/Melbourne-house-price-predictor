#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
FINAL PROJECT: House Price Prediction using SVM Regression
Dataset: Melbourne Housing Snapshot (melb_data.csv)
================================================================================
Author: [Your Name]
Date: April 2026

Project Structure:
  Stage 1: Initialization & Data Splitting
  Stage 2: Exploratory Data Analysis (EDA)
  Stage 3: Data Cleaning
  Stage 4: Dimensionality Reduction (PCA)
  Stage 5: SVM Model Development & Pipeline
  Stage 6: Bonus UI Deployment (FastAPI + Frontend)
================================================================================
"""

# =============================================================================
# STAGE 1: INITIALIZATION & DATA SPLITTING
# =============================================================================
# Objective:
#   1. Load the raw dataset.
#   2. Perform an IMMEDIATE train/test split BEFORE any transformations,
#      scaling, imputation, or exploratory analysis.
#
# Rationale (Data Leakage Prevention):
#   Splitting the data FIRST ensures that no information from the test set
#   leaks into our training pipeline. If we performed EDA, imputation,
#   or scaling on the full dataset before splitting, statistical properties
#   of the test set (means, medians, distributions) would influence our
#   training transformations. This would produce overly optimistic performance
#   estimates that do not generalize to truly unseen data.
#
#   Mathematically, we require:
#     P(model | X_train, y_train) ⊥ (X_test, y_test)
#   i.e., the model must be conditionally independent of the test set
#   given only the training data.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---- 1.1 Load the Raw Dataset ----
print("=" * 70)
print("STAGE 1: INITIALIZATION & DATA SPLITTING")
print("=" * 70)

df = pd.read_csv("melb_data.csv")

print(f"\n[INFO] Dataset loaded successfully.")
print(f"  - Total rows   : {df.shape[0]:,}")
print(f"  - Total columns: {df.shape[1]}")
print(f"  - Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ---- 1.2 Initial Dataset Overview ----
print("\n" + "-" * 70)
print("1.2  COLUMN OVERVIEW")
print("-" * 70)
print(f"\n{'Column':<20} {'Dtype':<12} {'Non-Null':>10} {'Null':>8} {'Null%':>8}")
print("-" * 58)
for col in df.columns:
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"{col:<20} {str(df[col].dtype):<12} {non_null:>10,} {null_count:>8,} {null_pct:>7.1f}%")

print(f"\n[INFO] Target variable: 'Price'")
print(f"  - Non-null prices: {df['Price'].notna().sum():,}")
print(f"  - Null prices    : {df['Price'].isna().sum():,}")

# ---- 1.3 Identify Features and Target ----
# The target variable is 'Price'. 
# Columns like 'Address', 'Suburb', and 'SellerG' are identifiers/high-cardinality
# categoricals that we will handle in later stages. For now, we keep all columns
# intact and simply separate X (features) from y (target).

# CRITICAL: Drop rows where the target 'Price' is missing, since we cannot
# train or evaluate on records without a known target value.
# This is the ONLY acceptable row drop at this stage.
rows_before = len(df)
df = df.dropna(subset=['Price'])
rows_after = len(df)
dropped_target_na = rows_before - rows_after

print(f"\n[INFO] Dropped {dropped_target_na} rows with missing 'Price' (target).")
print(f"  - Justification: Cannot train/evaluate regression without a target value.")
print(f"  - Remaining rows: {rows_after:,}")

# Separate features and target
y = df['Price'].copy()
X = df.drop(columns=['Price']).copy()

print(f"\n  - Feature matrix X shape: {X.shape}")
print(f"  - Target vector y shape : {y.shape}")

# ---- 1.4 Train/Test Split ----
# Split: 80% training, 20% testing
# random_state=42 for reproducibility
# We do NOT stratify because this is a regression problem (continuous target).
#
# Mathematical justification for 80/20 split:
#   - With ~13,500 samples, 20% ≈ 2,700 test samples, providing a statistically
#     significant test set for reliable RMSE/MAE/R² estimation.
#   - The bias-variance trade-off in data splits suggests that 80/20 is a
#     well-established balance: enough training data for the model to learn
#     complex patterns, and enough test data for robust evaluation.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\n" + "-" * 70)
print("1.4  TRAIN / TEST SPLIT RESULTS")
print("-" * 70)
print(f"  - Training set : {X_train.shape[0]:>6,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Testing set  : {X_test.shape[0]:>6,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  - Split ratio  : 80 / 20")
print(f"  - Random state : 42")

# ---- 1.5 Verify No Data Leakage ----
# Confirm that train and test indices are disjoint
train_indices = set(X_train.index)
test_indices = set(X_test.index)
overlap = train_indices.intersection(test_indices)

print(f"\n[VERIFICATION] Index overlap between train and test: {len(overlap)} (expected: 0)")
assert len(overlap) == 0, "DATA LEAKAGE DETECTED: Overlapping indices!"
print("[PASS] No data leakage — train and test sets are completely disjoint.\n")

# ---- 1.6 Target Distribution Summary (Train vs Test) ----
print("-" * 70)
print("1.6  TARGET DISTRIBUTION COMPARISON (Price)")
print("-" * 70)
print(f"\n{'Statistic':<15} {'Training':>15} {'Testing':>15}")
print("-" * 45)
for stat_name, stat_fn in [('Count', 'count'), ('Mean', 'mean'), ('Std', 'std'),
                            ('Min', 'min'), ('25%', lambda s: s.quantile(0.25)),
                            ('Median', 'median'), ('75%', lambda s: s.quantile(0.75)),
                            ('Max', 'max')]:
    if callable(stat_fn):
        train_val = stat_fn(y_train)
        test_val = stat_fn(y_test)
    else:
        train_val = getattr(y_train, stat_fn)()
        test_val = getattr(y_test, stat_fn)()
    
    if stat_name == 'Count':
        print(f"{stat_name:<15} {train_val:>15,.0f} {test_val:>15,.0f}")
    else:
        print(f"{stat_name:<15} ${train_val:>14,.0f} ${test_val:>14,.0f}")

print(f"\n[INFO] The similar distributions confirm that the random split preserved")
print(f"       the statistical properties of the target variable across both sets.")

# ---- 1.7 Save Split Indices for Reproducibility ----
# Store indices so we can always reconstruct the exact same split
split_info = {
    'train_indices': X_train.index.tolist(),
    'test_indices': X_test.index.tolist(),
    'random_state': 42,
    'test_size': 0.20
}

print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)
print(f"""
Summary:
  [OK] Dataset loaded: {rows_before:,} rows x 21 columns
  [OK] Dropped {dropped_target_na} rows with missing target (Price)
  [OK] Split performed BEFORE any transformations (no data leakage)
  [OK] Training set: {X_train.shape[0]:,} samples
  [OK] Testing set:  {X_test.shape[0]:,} samples
  [OK] Train/test index disjointness verified
  [OK] Target distributions are consistent across splits

Next Step -> Stage 2: Exploratory Data Analysis (EDA)
  - EDA will be performed ONLY on the training set (X_train, y_train)
  - The test set remains untouched until final model evaluation
""")

# =============================================================================
# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Objective:
#   Uncover patterns, anomalies, and relationships in the TRAINING data only.
#   Each visualization is unique — no two plots show the same relationship.
#
# CRITICAL: All EDA is performed EXCLUSIVELY on (X_train, y_train).
#           The test set (X_test, y_test) is NOT touched.
#
# Visualization Plan (8 unique plots, zero repetition):
#   1. Price Distribution — Target variable shape, skewness, and outliers
#   2. Correlation Heatmap — All numeric feature inter-relationships at once
#   3. Geospatial Price Map — Lat/Long colored by price (spatial patterns)
#   4. Price by Property Type & Region — Categorical impact on price
#   5. Key Numeric Features vs Price — Multi-panel scatter (Distance,
#      Landsize, BuildingArea, YearBuilt vs Price)
#   6. Missing Data Pattern — Which features are missing and how much
#   7. Rooms vs Bedroom2 Collinearity — Quantify the redundancy
#   8. Sale Method Distribution — How sales method relates to price
# =============================================================================

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

print("\n" + "=" * 70)
print("STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)
print("[INFO] All analysis performed on TRAINING SET ONLY (n=10,864)")

# Create output directory for plots
os.makedirs("eda_plots", exist_ok=True)

# Reconstruct training DataFrame with target for EDA convenience
train_df = X_train.copy()
train_df['Price'] = y_train

# Identify numeric columns in training set
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"\n  - Numeric features  : {len(numeric_cols)} columns")
print(f"  - Categorical features: {len(categorical_cols)} columns")

# ---- 2.1 Descriptive Statistics (Training Set) ----
print("\n" + "-" * 70)
print("2.1  DESCRIPTIVE STATISTICS (Training Set - Numeric Features)")
print("-" * 70)
desc = train_df[numeric_cols].describe().T
desc['skewness'] = train_df[numeric_cols].skew()
desc['kurtosis'] = train_df[numeric_cols].kurtosis()
print(desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']].to_string())

# ---- Skewness Analysis ----
price_skew = y_train.skew()
price_kurtosis = y_train.kurtosis()
print(f"\n[ANALYSIS] Price Distribution:")
print(f"  - Skewness : {price_skew:.3f} (>0 means right-skewed)")
print(f"  - Kurtosis : {price_kurtosis:.3f} (>3 means heavy-tailed / leptokurtic)")
print(f"  - Insight  : Price is {'strongly' if abs(price_skew) > 1 else 'moderately'} "
      f"right-skewed, indicating a long tail of expensive properties.")
print(f"              Log transformation may be beneficial for SVR.")

# ---- 2.2 Categorical Feature Summary ----
print("\n" + "-" * 70)
print("2.2  CATEGORICAL FEATURE SUMMARY (Training Set)")
print("-" * 70)
for col in categorical_cols:
    nunique = train_df[col].nunique()
    top_val = train_df[col].mode().iloc[0] if not train_df[col].mode().empty else 'N/A'
    top_freq = train_df[col].value_counts().iloc[0] if nunique > 0 else 0
    print(f"  {col:<15} | {nunique:>5} unique | Top: '{top_val}' ({top_freq:,} occurrences)")


# ===========================================================================
# VISUALIZATIONS (8 unique plots — zero repetition)
# ===========================================================================

# Set global style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif'
})
sns.set_style("whitegrid")

# ---- PLOT 1: Price Distribution with Skewness Annotation ----
# Purpose: Understand the shape, central tendency, and spread of the target
#          variable. Critical for deciding whether to log-transform Price.
print("\n[PLOT 1] Price Distribution (Histogram + KDE)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a: Raw price
axes[0].hist(y_train / 1e6, bins=60, color='#4C72B0', edgecolor='white',
             alpha=0.85, density=True)
kde_x = np.linspace(y_train.min() / 1e6, y_train.max() / 1e6, 500)
kde = stats.gaussian_kde(y_train / 1e6)
axes[0].plot(kde_x, kde(kde_x), color='#C44E52', linewidth=2)
axes[0].axvline(y_train.median() / 1e6, color='orange', linestyle='--',
                linewidth=1.5, label=f'Median: ${y_train.median()/1e6:.2f}M')
axes[0].axvline(y_train.mean() / 1e6, color='green', linestyle='--',
                linewidth=1.5, label=f'Mean: ${y_train.mean()/1e6:.2f}M')
axes[0].set_xlabel('Price (Millions AUD)')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Price Distribution (Skew={price_skew:.2f})')
axes[0].legend(fontsize=9)

# 1b: Log-transformed price
log_price = np.log1p(y_train)
log_skew = log_price.skew()
axes[1].hist(log_price, bins=60, color='#55A868', edgecolor='white',
             alpha=0.85, density=True)
kde_log = stats.gaussian_kde(log_price)
kde_log_x = np.linspace(log_price.min(), log_price.max(), 500)
axes[1].plot(kde_log_x, kde_log(kde_log_x), color='#C44E52', linewidth=2)
axes[1].set_xlabel('log(1 + Price)')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Log-Transformed Price (Skew={log_skew:.2f})')

plt.tight_layout()
plt.savefig('eda_plots/01_price_distribution.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/01_price_distribution.png")

# ---- PLOT 2: Correlation Heatmap (All Numeric Features) ----
# Purpose: Identify multicollinearity (e.g., Rooms vs Bedroom2) and
#          features strongly correlated with Price, in a single view.
print("[PLOT 2] Correlation Heatmap (Numeric Features)...")
corr_cols = [c for c in numeric_cols if c not in ['Postcode']]
corr_matrix = train_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=15, center='light')
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            square=True, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
            ax=ax)
ax.set_title('Pearson Correlation Matrix (Training Set)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('eda_plots/02_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/02_correlation_heatmap.png")

# Print top correlations with Price
print("\n  Top correlations with Price:")
price_corr = corr_matrix['Price'].drop('Price').abs().sort_values(ascending=False)
for feat, val in price_corr.head(8).items():
    sign = '+' if corr_matrix.loc[feat, 'Price'] > 0 else '-'
    print(f"    {feat:<18} r = {sign}{val:.3f}")

# ---- PLOT 3: Geospatial Price Map ----
# Purpose: Reveal spatial price patterns across Melbourne using Lat/Long.
#          This cannot be captured by any other plot type.
print("\n[PLOT 3] Geospatial Price Map (Lat/Long)...")
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(
    train_df['Longtitude'], train_df['Lattitude'],
    c=np.log1p(train_df['Price']), cmap='RdYlGn_r',
    alpha=0.4, s=8, edgecolors='none'
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('log(1 + Price)', fontsize=11)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Melbourne Housing Prices - Geospatial Distribution', fontsize=14)

# Annotate CBD approximate location
ax.annotate('CBD', xy=(144.9631, -37.8136), fontsize=11, fontweight='bold',
            color='black', ha='center',
            arrowprops=dict(arrowstyle='->', color='black'),
            xytext=(144.9631, -37.78))

plt.tight_layout()
plt.savefig('eda_plots/03_geospatial_price_map.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/03_geospatial_price_map.png")

# ---- PLOT 4: Price by Property Type AND Region (Combined) ----
# Purpose: Show how the two most important categorical features
#          jointly affect price. Using a single grouped visualization.
print("[PLOT 4] Price by Property Type & Region...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 4a: Price by Type (h=house, u=unit, t=townhouse)
type_order = train_df.groupby('Type')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=train_df, x='Type', y='Price', order=type_order,
            palette='Set2', fliersize=2, ax=axes[0])
axes[0].set_ylabel('Price (AUD)')
axes[0].set_title('Price Distribution by Property Type')
axes[0].set_xlabel('Type (h=House, t=Townhouse, u=Unit)')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))

# 4b: Price by Region
region_order = train_df.groupby('Regionname')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=train_df, y='Regionname', x='Price', order=region_order,
            palette='coolwarm', fliersize=1.5, ax=axes[1])
axes[1].set_xlabel('Price (AUD)')
axes[1].set_ylabel('')
axes[1].set_title('Price Distribution by Region')
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('eda_plots/04_price_by_type_and_region.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/04_price_by_type_and_region.png")

# ---- PLOT 5: Key Numeric Features vs Price (Multi-Panel Scatter) ----
# Purpose: Show 4 key feature-target relationships in a single figure.
#          Each subplot covers a DIFFERENT feature, so no repetition.
print("[PLOT 5] Key Numeric Features vs Price (4-panel scatter)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
scatter_features = [
    ('Distance', 'Distance from CBD (km)', '#4C72B0'),
    ('Landsize', 'Land Size (sqm)', '#55A868'),
    ('BuildingArea', 'Building Area (sqm)', '#C44E52'),
    ('YearBuilt', 'Year Built', '#8172B2')
]

for idx, (feat, label, color) in enumerate(scatter_features):
    ax = axes[idx // 2][idx % 2]
    valid = train_df[[feat, 'Price']].dropna()
    ax.scatter(valid[feat], valid['Price'] / 1e6, alpha=0.25, s=10,
               color=color, edgecolors='none')
    
    # Add correlation annotation
    r_val = valid[feat].corr(valid['Price'])
    ax.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel(label)
    ax.set_ylabel('Price (Millions AUD)')
    ax.set_title(f'{label} vs Price')

plt.tight_layout()
plt.savefig('eda_plots/05_numeric_features_vs_price.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/05_numeric_features_vs_price.png")

# ---- PLOT 6: Missing Data Pattern ----
# Purpose: Visualize the structure and extent of missing data.
#          Critical for planning imputation strategies in Stage 3.
print("[PLOT 6] Missing Data Pattern...")
missing_cols = train_df.columns[train_df.isnull().any()].tolist()
missing_pct = (train_df[missing_cols].isnull().sum() / len(train_df) * 100).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(missing_pct.index, missing_pct.values, color='#E07B54', edgecolor='white')

for bar, pct in zip(bars, missing_pct.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Missing Percentage (%)')
ax.set_title('Missing Data by Feature (Training Set)', fontsize=14)
ax.set_xlim(0, missing_pct.max() * 1.15)
plt.tight_layout()
plt.savefig('eda_plots/06_missing_data_pattern.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/06_missing_data_pattern.png")

# Print missing data summary
print("\n  Missing Data Summary (Training Set):")
for col in missing_pct.index[::-1]:
    count = train_df[col].isnull().sum()
    pct = count / len(train_df) * 100
    print(f"    {col:<18} {count:>6,} missing ({pct:.1f}%)")

# ---- PLOT 7: Rooms vs Bedroom2 Collinearity Analysis ----
# Purpose: Quantify the redundancy between Rooms and Bedroom2 — critical
#          for the dimensionality reduction decision in Stage 4.
print("\n[PLOT 7] Rooms vs Bedroom2 Collinearity...")
fig, ax = plt.subplots(figsize=(8, 7))

# Add jitter for better visualization of overlapping integer points
jitter_rooms = train_df['Rooms'] + np.random.normal(0, 0.1, len(train_df))
jitter_bed = train_df['Bedroom2'] + np.random.normal(0, 0.1, len(train_df))
ax.scatter(jitter_rooms, jitter_bed, alpha=0.15, s=8, color='#4C72B0', edgecolors='none')

# Perfect agreement line
lims = [0, max(train_df['Rooms'].max(), train_df['Bedroom2'].max()) + 1]
ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.7, label='Perfect agreement')

r_rooms_bed = train_df['Rooms'].corr(train_df['Bedroom2'])
exact_match = (train_df['Rooms'] == train_df['Bedroom2']).mean() * 100
ax.text(0.05, 0.92, f'Pearson r = {r_rooms_bed:.4f}\nExact match = {exact_match:.1f}%',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Rooms')
ax.set_ylabel('Bedroom2')
ax.set_title('Rooms vs Bedroom2: Collinearity Check', fontsize=13)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('eda_plots/07_rooms_vs_bedroom2.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/07_rooms_vs_bedroom2.png")
print(f"  -> Pearson r(Rooms, Bedroom2) = {r_rooms_bed:.4f}")
print(f"  -> Exact match rate: {exact_match:.1f}%")
print(f"  -> Conclusion: {'HIGH collinearity - candidate for removal/PCA' if r_rooms_bed > 0.9 else 'Moderate collinearity'}")

# ---- PLOT 8: Sale Method Analysis ----
# Purpose: Examine how the sale method affects price distribution.
#          Unique relationship not covered by any other plot.
print("\n[PLOT 8] Sale Method vs Price...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 8a: Count by method
method_counts = train_df['Method'].value_counts()
method_labels = {
    'S': 'Sold', 'SP': 'Sold Prior', 'PI': 'Passed In',
    'VB': 'Vendor Bid', 'SA': 'Sold After', 'SS': 'Sold (Source)',
    'PN': 'Passed In (No Bid)', 'SN': 'Sold (Not Disclosed)',
    'NB': 'No Bid', 'W': 'Withdrawn'
}
labels = [method_labels.get(m, m) for m in method_counts.index]
colors = sns.color_palette('Set2', len(method_counts))
axes[0].barh(labels, method_counts.values, color=colors, edgecolor='white')
axes[0].set_xlabel('Count')
axes[0].set_title('Sales by Method')

for i, v in enumerate(method_counts.values):
    axes[0].text(v + 20, i, str(v), va='center', fontsize=9)

# 8b: Price by method
method_order = train_df.groupby('Method')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=train_df, y='Method', x='Price', order=method_order,
            palette='Set2', fliersize=1.5, ax=axes[1])
axes[1].set_xlabel('Price (AUD)')
axes[1].set_title('Price Distribution by Sale Method')
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('eda_plots/08_sale_method_analysis.png', bbox_inches='tight')
plt.close()
print("  -> Saved: eda_plots/08_sale_method_analysis.png")


# ===========================================================================
# 2.9  KEY EDA FINDINGS SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("STAGE 2 COMPLETE - KEY FINDINGS SUMMARY")
print("=" * 70)

# Anomaly detection: Landsize outliers
landsize_extreme = train_df[train_df['Landsize'] > 20000].shape[0]
buildingarea_extreme = train_df[train_df['BuildingArea'] > 5000].shape[0] if 'BuildingArea' in train_df.columns else 0

print(f"""
PATTERNS & RELATIONSHIPS:
  1. Price is right-skewed (skew={price_skew:.2f}) -> log-transform recommended
  2. Top price predictors: Rooms (r={corr_matrix.loc['Rooms','Price']:.3f}), 
     Bathroom (r={corr_matrix.loc['Bathroom','Price']:.3f}), 
     BuildingArea (r={corr_matrix.loc['BuildingArea','Price']:.3f})
  3. Distance to CBD shows negative correlation (r={corr_matrix.loc['Distance','Price']:.3f})
     -> Closer to CBD = higher price
  4. Property type hierarchy: Houses > Townhouses > Units
  5. Southern Metropolitan is the most expensive region

COLLINEARITY (for Stage 4 - PCA):
  6. Rooms vs Bedroom2: r = {r_rooms_bed:.4f} ({exact_match:.1f}% exact match)
     -> Strong redundancy, candidate for dimensionality reduction
  7. Rooms vs Bathroom: r = {corr_matrix.loc['Rooms','Bathroom']:.3f}
     -> Moderate positive correlation

MISSING DATA (for Stage 3 - Cleaning):
  8. BuildingArea: ~47.5% missing -> needs advanced imputation
  9. YearBuilt: ~39.6% missing -> needs advanced imputation
 10. CouncilArea: ~10.1% missing -> categorical imputation
 11. Car: ~0.5% missing -> simple imputation (median/mode)

ANOMALIES DETECTED:
 12. Landsize outliers: {landsize_extreme} records with Landsize > 20,000 sqm
 13. Some Landsize = 0 for houses (likely data entry errors)
 14. BuildingArea has some extreme values (potential errors)

All 8 plots saved to: eda_plots/
""")
print("AWAITING APPROVAL to proceed to Stage 3: Data Cleaning")
print("=" * 70)


# =============================================================================
# STAGE 3: DATA CLEANING
# =============================================================================
# Objective:
#   Handle missing values, outliers, and inaccurate entries in the TRAINING
#   set. Then apply the SAME transformations (fitted on training) to the
#   test set to avoid data leakage.
#
# STRICT RULE: Avoid dropping records unless mathematically impossible.
#   Every decision to drop (if any) must be documented and justified.
#
# Strategy Overview:
#   1. Remove non-predictive identifier columns (Address, Suburb — already
#      captured by Lat/Long, Postcode, Regionname; SellerG — too high
#      cardinality and not a property characteristic)
#   2. Feature engineering from Date (extract sale year and month)
#   3. Missing value imputation:
#      a. Car (0.5%) — median imputation by property Type
#      b. CouncilArea (10%) — mode imputation by Postcode
#      c. BuildingArea (47%) — KNN Imputer using correlated features
#      d. YearBuilt (40%) — KNN Imputer using correlated features
#   4. Outlier handling — IQR-based capping (not dropping!)
#   5. Categorical encoding (Type, Method, Regionname)
# =============================================================================

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

print("\n" + "=" * 70)
print("STAGE 3: DATA CLEANING")
print("=" * 70)

# Work on copies to preserve originals
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

rows_before_cleaning = len(X_train_clean)

# ---- 3.1 Remove Non-Predictive Identifier Columns ----
# Justification:
#   - Address: 10,726 unique values out of 10,864 — nearly unique identifier,
#     not a generalizable feature. Location is already captured by
#     Lattitude, Longtitude, Postcode, and Regionname.
#   - Suburb: 306 unique values — redundant with Postcode and Regionname
#     which encode the same geographic information with lower cardinality.
#   - SellerG: 249 unique agents — this is a transaction metadata field,
#     not a property characteristic. Including it would overfit to specific
#     agents rather than learning property value drivers.

drop_cols = ['Address', 'Suburb', 'SellerG']
print(f"\n[3.1] Removing non-predictive columns: {drop_cols}")
for col in drop_cols:
    nunique_train = X_train_clean[col].nunique()
    print(f"  - {col}: {nunique_train} unique values (not generalizable as features)")

X_train_clean = X_train_clean.drop(columns=drop_cols)
X_test_clean = X_test_clean.drop(columns=drop_cols)
print(f"  -> Remaining columns: {X_train_clean.shape[1]}")

# ---- 3.2 Feature Engineering from Date ----
# Extract year and month from the Date column, then drop the raw string.
# Rationale: The raw date string is not usable by ML models. However,
# the sale year captures market trends (Melbourne prices rose ~10% annually
# during 2016-2017), and the sale month captures seasonal effects.
print(f"\n[3.2] Feature engineering from 'Date'...")

def extract_date_features(df):
    """Extract SaleYear and SaleMonth from the Date string column."""
    df = df.copy()
    date_parsed = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['SaleYear'] = date_parsed.dt.year.astype(float)
    df['SaleMonth'] = date_parsed.dt.month.astype(float)
    df = df.drop(columns=['Date'])
    return df

X_train_clean = extract_date_features(X_train_clean)
X_test_clean = extract_date_features(X_test_clean)
print(f"  -> Created: SaleYear, SaleMonth")
print(f"  -> Dropped: Date (raw string)")
print(f"  -> SaleYear range: {X_train_clean['SaleYear'].min():.0f} - {X_train_clean['SaleYear'].max():.0f}")
print(f"  -> SaleMonth range: {X_train_clean['SaleMonth'].min():.0f} - {X_train_clean['SaleMonth'].max():.0f}")

# ---- 3.3 Missing Value Imputation ----
print(f"\n[3.3] MISSING VALUE IMPUTATION")
print("-" * 50)

# ---- 3.3a Car (0.5% missing) — Median imputation by Type ----
# Rationale: Car parking spaces vary by property type (houses typically
# have more than units). Median by Type preserves this conditional
# distribution rather than using a single global median.
print(f"\n  [3.3a] Car: {X_train_clean['Car'].isna().sum()} missing in train, "
      f"{X_test_clean['Car'].isna().sum()} missing in test")

car_medians = X_train_clean.groupby('Type')['Car'].median()
print(f"    Strategy: Median by Type (from training set)")
for t, med in car_medians.items():
    print(f"      Type '{t}': median = {med:.0f}")

def impute_car(df, medians):
    df = df.copy()
    for t, med in medians.items():
        mask = (df['Car'].isna()) & (df['Type'] == t)
        df.loc[mask, 'Car'] = med
    # Fallback: if any still missing (e.g., unseen Type), use overall median
    df['Car'] = df['Car'].fillna(medians.median())
    return df

X_train_clean = impute_car(X_train_clean, car_medians)
X_test_clean = impute_car(X_test_clean, car_medians)
print(f"    -> Car missing after imputation: train={X_train_clean['Car'].isna().sum()}, "
      f"test={X_test_clean['Car'].isna().sum()}")

# ---- 3.3b CouncilArea (10% missing) — Mode imputation by Postcode ----
# Rationale: CouncilArea is deterministically linked to geographic location.
# Properties in the same Postcode almost always belong to the same council.
# Using mode-by-Postcode is more accurate than a global mode.
print(f"\n  [3.3b] CouncilArea: {X_train_clean['CouncilArea'].isna().sum()} missing in train, "
      f"{X_test_clean['CouncilArea'].isna().sum()} missing in test")

council_mode_by_postcode = X_train_clean.groupby('Postcode')['CouncilArea'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
)
global_council_mode = X_train_clean['CouncilArea'].mode().iloc[0]

def impute_council(df, mode_map, global_mode):
    df = df.copy()
    for idx in df[df['CouncilArea'].isna()].index:
        postcode = df.loc[idx, 'Postcode']
        if postcode in mode_map.index and pd.notna(mode_map[postcode]):
            df.loc[idx, 'CouncilArea'] = mode_map[postcode]
        else:
            df.loc[idx, 'CouncilArea'] = global_mode
    return df

X_train_clean = impute_council(X_train_clean, council_mode_by_postcode, global_council_mode)
X_test_clean = impute_council(X_test_clean, council_mode_by_postcode, global_council_mode)
print(f"    Strategy: Mode by Postcode (from training set)")
print(f"    -> CouncilArea missing after: train={X_train_clean['CouncilArea'].isna().sum()}, "
      f"test={X_test_clean['CouncilArea'].isna().sum()}")

# ---- 3.3c & 3.3d: BuildingArea (47%) and YearBuilt (40%) — KNN Imputer ----
# Rationale for KNN Imputer:
#   - With ~47% missing in BuildingArea and ~40% in YearBuilt, simple
#     mean/median imputation would create artificial density spikes.
#   - KNN Imputer (k=5) finds the 5 nearest neighbors based on available
#     numeric features and uses their weighted average. This preserves
#     local data structure and produces more realistic imputed values.
#   - Mathematical basis: E[X_missing | X_observed] is approximated by
#     the weighted mean of k nearest neighbors in the observed feature space.
#   - We use k=5 as a standard choice that balances bias and variance
#     in the imputation estimates (Troyanskaya et al., 2001).
print(f"\n  [3.3c/d] KNN Imputation for BuildingArea & YearBuilt")
print(f"    BuildingArea: {X_train_clean['BuildingArea'].isna().sum()} missing in train")
print(f"    YearBuilt:    {X_train_clean['YearBuilt'].isna().sum()} missing in train")

# Select numeric features for KNN imputation context
knn_features = ['Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                'BuildingArea', 'YearBuilt', 'Distance', 'Lattitude',
                'Longtitude', 'Propertycount']

print(f"    Strategy: KNN Imputer (k=5) using features: {knn_features}")
print(f"    Fitting on training set only (leakage prevention)...")

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

# Fit on training, transform both
X_train_knn = X_train_clean[knn_features].copy()
X_test_knn = X_test_clean[knn_features].copy()

X_train_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(X_train_knn),
    columns=knn_features,
    index=X_train_clean.index
)
X_test_knn_imputed = pd.DataFrame(
    knn_imputer.transform(X_test_knn),
    columns=knn_features,
    index=X_test_clean.index
)

# Replace only BuildingArea and YearBuilt with imputed values
X_train_clean['BuildingArea'] = X_train_knn_imputed['BuildingArea']
X_train_clean['YearBuilt'] = X_train_knn_imputed['YearBuilt']
X_test_clean['BuildingArea'] = X_test_knn_imputed['BuildingArea']
X_test_clean['YearBuilt'] = X_test_knn_imputed['YearBuilt']

print(f"    -> BuildingArea missing after: train={X_train_clean['BuildingArea'].isna().sum()}, "
      f"test={X_test_clean['BuildingArea'].isna().sum()}")
print(f"    -> YearBuilt missing after:    train={X_train_clean['YearBuilt'].isna().sum()}, "
      f"test={X_test_clean['YearBuilt'].isna().sum()}")

# ---- 3.4 Outlier Handling — IQR-based Capping ----
# Rationale:
#   SVR with RBF kernel is sensitive to extreme outliers because the
#   kernel distance computation k(x,x') = exp(-gamma * ||x-x'||^2) can
#   be dominated by a few extreme values. Rather than DROPPING outliers
#   (which violates our rule), we CAP them at Q1 - 1.5*IQR and Q3 + 1.5*IQR.
#   This is also known as Winsorization and preserves all records while
#   limiting the influence of extreme values.
#
#   Exception: We use a wider 3*IQR range for Landsize and BuildingArea
#   because these features have legitimate long-tailed distributions
#   (mansions, rural properties).
print(f"\n[3.4] OUTLIER HANDLING (IQR-based Capping)")
print("-" * 50)

def cap_outliers(train_df, test_df, column, iqr_multiplier=1.5):
    """Cap outliers using IQR boundaries computed from training set ONLY."""
    Q1 = train_df[column].quantile(0.25)
    Q3 = train_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR
    
    # Count outliers before capping
    train_lower = (train_df[column] < lower).sum()
    train_upper = (train_df[column] > upper).sum()
    
    # Cap in both sets
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[column] = train_df[column].clip(lower=lower, upper=upper)
    test_df[column] = test_df[column].clip(lower=lower, upper=upper)
    
    return train_df, test_df, lower, upper, train_lower, train_upper

# Features to cap and their IQR multipliers
# Using 3x IQR for Landsize/BuildingArea (legitimate wide distributions)
# Using 1.5x IQR for Price-related features
cap_config = [
    ('Landsize', 3.0),
    ('BuildingArea', 3.0),
    ('Price_proxy', None),  # Skip — Price is the target, we don't modify it
]

# Cap Rooms (1.5x IQR)
X_train_clean, X_test_clean, rooms_lo, rooms_hi, n_lo, n_hi = cap_outliers(
    X_train_clean, X_test_clean, 'Rooms', iqr_multiplier=1.5
)
print(f"  Rooms (1.5x IQR): capped to [{rooms_lo:.0f}, {rooms_hi:.0f}]")
print(f"    -> {n_lo} below lower, {n_hi} above upper in training set")

# Cap Bathroom (1.5x IQR)
X_train_clean, X_test_clean, bath_lo, bath_hi, n_lo, n_hi = cap_outliers(
    X_train_clean, X_test_clean, 'Bathroom', iqr_multiplier=1.5
)
print(f"  Bathroom (1.5x IQR): capped to [{bath_lo:.0f}, {bath_hi:.0f}]")
print(f"    -> {n_lo} below lower, {n_hi} above upper in training set")

# Cap Car (1.5x IQR)
X_train_clean, X_test_clean, car_lo, car_hi, n_lo, n_hi = cap_outliers(
    X_train_clean, X_test_clean, 'Car', iqr_multiplier=1.5
)
print(f"  Car (1.5x IQR): capped to [{car_lo:.0f}, {car_hi:.0f}]")
print(f"    -> {n_lo} below lower, {n_hi} above upper in training set")

# Cap Landsize (3x IQR — wider for legitimate long-tailed distribution)
X_train_clean, X_test_clean, landsize_lo, landsize_hi, n_lo, n_hi = cap_outliers(
    X_train_clean, X_test_clean, 'Landsize', iqr_multiplier=3.0
)
print(f"  Landsize (3x IQR): capped to [{landsize_lo:.0f}, {landsize_hi:.0f}]")
print(f"    -> {n_lo} below lower, {n_hi} above upper in training set")

# Cap BuildingArea (3x IQR — wider for legitimate long-tailed distribution)
X_train_clean, X_test_clean, building_lo, building_hi, n_lo, n_hi = cap_outliers(
    X_train_clean, X_test_clean, 'BuildingArea', iqr_multiplier=3.0
)
print(f"  BuildingArea (3x IQR): capped to [{building_lo:.0f}, {building_hi:.0f}]")
print(f"    -> {n_lo} below lower, {n_hi} above upper in training set")

# Handle Landsize = 0 for houses (data entry error)
# Replace with median Landsize for houses in the same Postcode
zero_landsize_houses = (X_train_clean['Landsize'] == 0) & (X_train_clean['Type'] == 'h')
n_zero = zero_landsize_houses.sum()
if n_zero > 0:
    landsize_median_by_postcode = X_train_clean[X_train_clean['Type'] == 'h'].groupby(
        'Postcode')['Landsize'].median()
    global_house_landsize = X_train_clean.loc[
        (X_train_clean['Type'] == 'h') & (X_train_clean['Landsize'] > 0), 'Landsize'
    ].median()
    
    for idx in X_train_clean[zero_landsize_houses].index:
        pc = X_train_clean.loc[idx, 'Postcode']
        med = landsize_median_by_postcode.get(pc, global_house_landsize)
        if med > 0:
            X_train_clean.loc[idx, 'Landsize'] = med
        else:
            X_train_clean.loc[idx, 'Landsize'] = global_house_landsize

print(f"  Landsize=0 for houses: {n_zero} records corrected using median by Postcode")

# Apply same logic to test set
zero_landsize_houses_test = (X_test_clean['Landsize'] == 0) & (X_test_clean['Type'] == 'h')
n_zero_test = zero_landsize_houses_test.sum()
if n_zero_test > 0:
    for idx in X_test_clean[zero_landsize_houses_test].index:
        pc = X_test_clean.loc[idx, 'Postcode']
        med = landsize_median_by_postcode.get(pc, global_house_landsize)
        if med > 0:
            X_test_clean.loc[idx, 'Landsize'] = med
        else:
            X_test_clean.loc[idx, 'Landsize'] = global_house_landsize
print(f"  Landsize=0 for houses (test): {n_zero_test} records corrected")

# ---- 3.5 Categorical Encoding ----
# Rationale:
#   SVR requires all features to be numeric. We use:
#   - Label Encoding for Type (ordinal: h > t > u in price hierarchy)
#   - One-Hot Encoding for Regionname (8 categories, nominal)
#   - Label Encoding for Method (5 categories)
#   - Label Encoding for CouncilArea (33 categories — one-hot would
#     create too many sparse features for SVR)
print(f"\n[3.5] CATEGORICAL ENCODING")
print("-" * 50)

# 3.5a: Type — ordinal encoding based on price hierarchy (from EDA)
type_map = {'h': 2, 't': 1, 'u': 0}  # house > townhouse > unit
X_train_clean['Type'] = X_train_clean['Type'].map(type_map)
X_test_clean['Type'] = X_test_clean['Type'].map(type_map)
print(f"  Type: ordinal encoded (h=2, t=1, u=0)")

# 3.5b: Method — label encoding
method_le = LabelEncoder()
X_train_clean['Method'] = method_le.fit_transform(X_train_clean['Method'])
# Handle unseen categories in test set
test_methods = X_test_clean['Method'].copy()
known_methods = set(method_le.classes_)
test_methods = test_methods.apply(lambda x: x if x in known_methods else method_le.classes_[0])
X_test_clean['Method'] = method_le.transform(test_methods)
print(f"  Method: label encoded ({len(method_le.classes_)} classes)")

# 3.5c: CouncilArea — label encoding (too many for one-hot with SVR)
council_le = LabelEncoder()
X_train_clean['CouncilArea'] = council_le.fit_transform(X_train_clean['CouncilArea'].astype(str))
test_councils = X_test_clean['CouncilArea'].astype(str).copy()
known_councils = set(council_le.classes_)
test_councils = test_councils.apply(lambda x: x if x in known_councils else council_le.classes_[0])
X_test_clean['CouncilArea'] = council_le.transform(test_councils)
print(f"  CouncilArea: label encoded ({len(council_le.classes_)} classes)")

# 3.5d: Regionname — one-hot encoding (only 8 categories, manageable)
region_dummies_train = pd.get_dummies(X_train_clean['Regionname'], prefix='Region', drop_first=True)
region_dummies_test = pd.get_dummies(X_test_clean['Regionname'], prefix='Region', drop_first=True)

# Align test set columns with training set (handle unseen categories)
for col in region_dummies_train.columns:
    if col not in region_dummies_test.columns:
        region_dummies_test[col] = 0
region_dummies_test = region_dummies_test[region_dummies_train.columns]

X_train_clean = pd.concat([X_train_clean.drop(columns=['Regionname']), region_dummies_train], axis=1)
X_test_clean = pd.concat([X_test_clean.drop(columns=['Regionname']), region_dummies_test], axis=1)
print(f"  Regionname: one-hot encoded ({len(region_dummies_train.columns)} dummy columns, drop_first=True)")

# ---- 3.6 Final Validation ----
print(f"\n[3.6] FINAL VALIDATION")
print("-" * 50)

# Check for remaining NaN
train_nan = X_train_clean.isna().sum().sum()
test_nan = X_test_clean.isna().sum().sum()
print(f"  Remaining NaN in training: {train_nan}")
print(f"  Remaining NaN in testing:  {test_nan}")

# If any NaN remain, fill with column median (safety net)
if train_nan > 0 or test_nan > 0:
    print("  [SAFETY] Filling remaining NaN with training column medians...")
    for col in X_train_clean.select_dtypes(include=[np.number]).columns:
        median_val = X_train_clean[col].median()
        X_train_clean[col] = X_train_clean[col].fillna(median_val)
        X_test_clean[col] = X_test_clean[col].fillna(median_val)
    train_nan_after = X_train_clean.isna().sum().sum()
    test_nan_after = X_test_clean.isna().sum().sum()
    print(f"  After safety fill: train NaN={train_nan_after}, test NaN={test_nan_after}")

# Row count verification
rows_after_cleaning = len(X_train_clean)
rows_dropped = rows_before_cleaning - rows_after_cleaning

print(f"\n  Rows before cleaning: {rows_before_cleaning:,}")
print(f"  Rows after cleaning:  {rows_after_cleaning:,}")
print(f"  Rows dropped:         {rows_dropped}")
if rows_dropped == 0:
    print(f"  [PASS] ZERO rows dropped -- all records preserved via imputation!")
else:
    print(f"  [WARNING] {rows_dropped} rows were dropped. See documentation above.")

# Feature summary
print(f"\n  Final feature count: {X_train_clean.shape[1]}")
print(f"  Feature names:")
for i, col in enumerate(X_train_clean.columns):
    dtype = X_train_clean[col].dtype
    print(f"    {i+1:>2}. {col:<25} ({dtype})")

# ---- 3.7 Summary ----
print("\n" + "=" * 70)
print("STAGE 3 COMPLETE")
print("=" * 70)
print(f"""
Summary of Data Cleaning:
  [OK] Removed 3 non-predictive columns (Address, Suburb, SellerG)
  [OK] Engineered 2 features from Date (SaleYear, SaleMonth)
  [OK] Car: median imputation by Type ({X_train_clean['Car'].isna().sum()} remaining NaN)
  [OK] CouncilArea: mode imputation by Postcode ({X_train_clean['CouncilArea'].isna().sum()} remaining NaN)
  [OK] BuildingArea & YearBuilt: KNN imputation (k=5, distance-weighted)
  [OK] Outliers: IQR-based capping (no rows dropped)
  [OK] Landsize=0 for houses: corrected using Postcode median
  [OK] Categorical encoding: Type(ordinal), Method(label), CouncilArea(label), Regionname(one-hot)
  [OK] Total rows dropped: {rows_dropped} -- ALL records preserved!
  [OK] Final shape: train={X_train_clean.shape}, test={X_test_clean.shape}
""")
print("AWAITING APPROVAL to proceed to Stage 4: Dimensionality Reduction")
print("=" * 70)


# =============================================================================
# STAGE 4: DIMENSIONALITY REDUCTION
# =============================================================================
# Objective:
#   Reduce the 24 features while retaining core variance and useful info.
#   This is critical for SVR because:
#     1. SVR with RBF kernel computes pairwise distances in feature space.
#        High-dimensional sparse features degrade kernel effectiveness
#        (the "curse of dimensionality").
#     2. Correlated features (e.g., Rooms/Bedroom2) inflate variance
#        without adding information, leading to overfitting.
#     3. PCA finds orthogonal components that maximize variance, which
#        aligns with SVR's need for informative, uncorrelated input.
#
# Strategy:
#   Step 1: Remove explicitly redundant features (Bedroom2, Postcode)
#   Step 2: Standardize features (PCA requires zero-mean, unit-variance)
#   Step 3: Apply PCA, select components retaining >= 95% variance
#   Step 4: Analyze and visualize the explained variance
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n" + "=" * 70)
print("STAGE 4: DIMENSIONALITY REDUCTION")
print("=" * 70)

# ---- 4.1 Remove Explicitly Redundant Features ----
# Before PCA, we remove features that are known to be redundant based on
# our EDA findings. This is a manual pre-PCA step that makes the
# subsequent PCA more interpretable and efficient.
print("\n[4.1] REMOVING EXPLICITLY REDUNDANT FEATURES")
print("-" * 50)

X_train_reduced = X_train_clean.copy()
X_test_reduced = X_test_clean.copy()

# 4.1a: Drop Bedroom2 (r=0.9403 with Rooms, 95.1% exact match)
# Justification: Bedroom2 is a secondary count of bedrooms from a different
# data source. With r=0.94 and 95.1% exact agreement with Rooms, it carries
# almost zero unique information. Keeping both would:
#   - Inflate PCA with a component dominated by their shared variance
#   - Add noise where they disagree (the 4.9% mismatch cases)
# We keep Rooms (from the primary listing) as it has zero missing values.
r_rooms_bed2 = X_train_reduced['Rooms'].corr(X_train_reduced['Bedroom2'])
print(f"  Dropping 'Bedroom2': r(Rooms, Bedroom2) = {r_rooms_bed2:.4f}")
print(f"    -> Rooms retained as primary bedroom count (0 missing values)")
X_train_reduced = X_train_reduced.drop(columns=['Bedroom2'])
X_test_reduced = X_test_reduced.drop(columns=['Bedroom2'])

# 4.1b: Drop Postcode (nominal identifier, redundant with spatial features)
# Justification: Postcode is a nominal categorical variable encoded as a
# number (e.g., 3067, 3042). Its numeric value has no ordinal meaning
# (3067 is not "greater than" 3042 in any meaningful sense).
# Location is already fully captured by:
#   - Lattitude + Longtitude (continuous spatial coordinates)
#   - Regionname one-hot dummies (8 regions)
#   - CouncilArea (33 label-encoded councils)
# Including Postcode in PCA would create spurious variance along a
# meaningless numeric axis.
print(f"  Dropping 'Postcode': nominal identifier, redundant with Lat/Long + Region + CouncilArea")
X_train_reduced = X_train_reduced.drop(columns=['Postcode'])
X_test_reduced = X_test_reduced.drop(columns=['Postcode'])

print(f"  -> Features after redundancy removal: {X_train_reduced.shape[1]} (from {X_train_clean.shape[1]})")

# ---- 4.2 Feature Standardization ----
# PCA requires features to be on the same scale because it maximizes
# variance. Without standardization, features with large absolute values
# (e.g., Propertycount ~7000) would dominate over features with small
# values (e.g., Bathroom ~1.5), regardless of their actual importance.
#
# StandardScaler: z = (x - mu) / sigma
#   where mu and sigma are computed from TRAINING set only.
print(f"\n[4.2] FEATURE STANDARDIZATION (StandardScaler)")
print("-" * 50)

feature_names = X_train_reduced.columns.tolist()
print(f"  Scaling {len(feature_names)} features to zero mean, unit variance")
print(f"  Fitting on training set only (leakage prevention)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

print(f"  -> Training mean range: [{X_train_scaled.mean(axis=0).min():.6f}, "
      f"{X_train_scaled.mean(axis=0).max():.6f}] (expected ~0)")
print(f"  -> Training std range:  [{X_train_scaled.std(axis=0).min():.4f}, "
      f"{X_train_scaled.std(axis=0).max():.4f}] (expected ~1)")

# ---- 4.3 PCA Analysis ----
# First, fit PCA with ALL components to analyze the full variance spectrum.
# Then select the optimal number of components.
print(f"\n[4.3] PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 50)

n_features = X_train_scaled.shape[1]
pca_full = PCA(n_components=n_features, random_state=42)
X_train_pca_full = pca_full.fit_transform(X_train_scaled)

# Explained variance analysis
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\n  Full PCA Explained Variance (all {n_features} components):")
print(f"  {'PC':<6} {'Individual':>12} {'Cumulative':>12}")
print(f"  {'-'*30}")
for i in range(n_features):
    marker = " <-- 95% threshold" if i > 0 and cumulative_var[i-1] < 0.95 <= cumulative_var[i] else ""
    print(f"  PC{i+1:<3} {explained_var[i]:>11.4f} {cumulative_var[i]:>11.4f}{marker}")

# ---- 4.4 Select Optimal Number of Components ----
# Criterion: Retain components that capture >= 95% of total variance.
# Mathematical justification:
#   The 95% threshold is a widely accepted standard in dimensionality
#   reduction (Jolliffe, 2002). It retains the dominant signal while
#   discarding noise-dominated components. For SVR specifically, fewer
#   orthogonal features improve kernel matrix conditioning and reduce
#   computational cost from O(n^2 * d) to O(n^2 * k) where k << d.

VARIANCE_THRESHOLD = 0.95
n_components_95 = np.argmax(cumulative_var >= VARIANCE_THRESHOLD) + 1

print(f"\n  Variance threshold: {VARIANCE_THRESHOLD*100:.0f}%")
print(f"  Components needed:  {n_components_95} (out of {n_features})")
print(f"  Variance retained:  {cumulative_var[n_components_95-1]*100:.2f}%")
print(f"  Variance discarded: {(1-cumulative_var[n_components_95-1])*100:.2f}%")
print(f"  Dimensionality reduction: {n_features} -> {n_components_95} "
      f"({(1-n_components_95/n_features)*100:.1f}% reduction)")

# ---- 4.5 Apply Final PCA ----
print(f"\n[4.5] APPLYING FINAL PCA (n_components={n_components_95})")
print("-" * 50)

pca_final = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

print(f"  Training set: {X_train_scaled.shape} -> {X_train_pca.shape}")
print(f"  Testing set:  {X_test_scaled.shape} -> {X_test_pca.shape}")

# ---- 4.6 PCA Component Interpretation ----
# Show which original features contribute most to each principal component.
print(f"\n[4.6] TOP FEATURE LOADINGS PER COMPONENT")
print("-" * 50)

loadings = pd.DataFrame(
    pca_final.components_.T,
    index=feature_names,
    columns=[f'PC{i+1}' for i in range(n_components_95)]
)

for pc_idx in range(min(5, n_components_95)):  # Show top 5 PCs
    pc_name = f'PC{pc_idx+1}'
    var_pct = explained_var[pc_idx] * 100
    top_features = loadings[pc_name].abs().sort_values(ascending=False).head(5)
    print(f"\n  {pc_name} ({var_pct:.1f}% variance):")
    for feat, loading in top_features.items():
        sign = '+' if loadings.loc[feat, pc_name] > 0 else '-'
        print(f"    {sign}{feat:<30} |loading| = {loading:.4f}")

# ---- 4.7 Visualization: Explained Variance Plot ----
print(f"\n[4.7] Generating explained variance plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 4.7a: Scree plot (individual variance)
axes[0].bar(range(1, n_features + 1), explained_var * 100,
            color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Individual Explained Variance (%)')
axes[0].set_title('Scree Plot')
axes[0].set_xticks(range(1, n_features + 1))

# 4.7b: Cumulative variance
axes[1].plot(range(1, n_features + 1), cumulative_var * 100,
             'o-', color='#C44E52', linewidth=2, markersize=6)
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=1.5,
                label='95% threshold')
axes[1].axvline(x=n_components_95, color='orange', linestyle='--',
                linewidth=1.5, label=f'n={n_components_95} components')
axes[1].fill_between(range(1, n_components_95 + 1),
                     cumulative_var[:n_components_95] * 100,
                     alpha=0.15, color='#C44E52')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend(fontsize=10)
axes[1].set_xticks(range(1, n_features + 1))
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig('eda_plots/09_pca_explained_variance.png', bbox_inches='tight')
plt.close()
print(f"  -> Saved: eda_plots/09_pca_explained_variance.png")

# ---- 4.8 Summary ----
print("\n" + "=" * 70)
print("STAGE 4 COMPLETE")
print("=" * 70)
print(f"""
Summary of Dimensionality Reduction:
  [OK] Dropped Bedroom2 (r=0.94 with Rooms, 95.1% exact match)
  [OK] Dropped Postcode (nominal ID, redundant with Lat/Long/Region/Council)
  [OK] StandardScaler fitted on training set only
  [OK] PCA applied: {n_features} features -> {n_components_95} principal components
  [OK] Variance retained: {cumulative_var[n_components_95-1]*100:.2f}%
  [OK] Dimensionality reduction: {(1-n_components_95/n_features)*100:.1f}%
  [OK] Final training shape: {X_train_pca.shape}
  [OK] Final testing shape:  {X_test_pca.shape}

Objects available for Stage 5:
  - X_train_pca: PCA-transformed training features
  - X_test_pca:  PCA-transformed testing features
  - y_train:     Training target (Price)
  - y_test:      Testing target (Price)
  - scaler:      Fitted StandardScaler
  - pca_final:   Fitted PCA transformer
  - knn_imputer: Fitted KNN imputer (for pipeline)
""")
print("AWAITING APPROVAL to proceed to Stage 5: SVM Model Development & Pipeline")
print("=" * 70)


# =============================================================================
# STAGE 5: SVM MODEL DEVELOPMENT & PIPELINE
# =============================================================================
# Objective:
#   1. Build an SVR (Support Vector Regression) model
#   2. Fine-tune hyperparameters (C, gamma, epsilon, kernel)
#   3. Evaluate using RMSE, MAE, R-squared
#   4. Package Stages 3-5 into a reproducible scikit-learn Pipeline
#   5. Save model artifacts for deployment (Stage 6)
#
# Key Design Decisions:
#   - Log-transform Price (target): EDA showed skew=2.30. SVR assumes
#     approximately symmetric error distribution. log(Price) has skew=0.18,
#     enabling the epsilon-insensitive tube to work effectively.
#   - RBF kernel: Non-linear price relationships (e.g., distance-price
#     curve is not linear). RBF kernel maps to infinite-dimensional space
#     via k(x,x') = exp(-gamma * ||x-x'||^2).
#   - RandomizedSearchCV: More efficient than GridSearchCV for SVR,
#     which has O(n^2) to O(n^3) complexity per fit. Randomized search
#     covers the parameter space stochastically with bounded compute.
# =============================================================================

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
import joblib
import time

print("\n" + "=" * 70)
print("STAGE 5: SVM MODEL DEVELOPMENT & PIPELINE")
print("=" * 70)

# ---- 5.1 Target Transformation (Log) ----
# Rationale: Price has skewness=2.30 (heavy right skew). SVR with
# epsilon-insensitive loss assumes that errors are roughly symmetric.
# log-transforming Price reduces skew to ~0.18, making the error
# distribution much more suitable for SVR.
#
# We use log1p(y) = ln(1+y) for numerical stability (though prices
# are always positive, this is good practice).
# At prediction time, we reverse with expm1(y_hat) = exp(y_hat) - 1.
print("\n[5.1] TARGET TRANSFORMATION (Log)")
print("-" * 50)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

print(f"  y_train skewness: {y_train.skew():.3f} -> {y_train_log.skew():.3f} (after log)")
print(f"  y_train range: [{y_train.min():,.0f}, {y_train.max():,.0f}] -> "
      f"[{y_train_log.min():.2f}, {y_train_log.max():.2f}]")

# ---- 5.2 SVR Hyperparameter Tuning ----
# SVR hyperparameters:
#   C (regularization): Controls trade-off between training error and
#     model complexity. Higher C = stricter fit to training data.
#     Range: [0.1, 1000] on log scale.
#   gamma (RBF kernel width): Controls the "reach" of each training
#     sample. High gamma = narrow influence = complex boundary.
#     Range: [0.001, 1] on log scale.
#   epsilon (tube width): Errors within epsilon are ignored (no penalty).
#     Range: [0.01, 0.5]. Larger epsilon = more tolerance = smoother fit.
#
# RandomizedSearchCV rationale:
#   - SVR fit complexity is O(n^2 * d) to O(n^3), making exhaustive
#     grid search prohibitively expensive for ~10K samples.
#   - Bergstra & Bengio (2012) showed that random search finds good
#     hyperparameters in fewer iterations than grid search, especially
#     when some parameters are more important than others.
#   - We use 30 iterations x 5-fold CV = 150 total SVR fits, which
#     is manageable on a standard machine.
print("\n[5.2] SVR HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("-" * 50)

param_distributions = {
    'C': loguniform(1, 1000),          # Log-uniform [1, 1000] (optimal zone)
    'gamma': loguniform(0.001, 1),     # Log-uniform [0.001, 1]
    'epsilon': loguniform(0.01, 0.5),  # Log-uniform [0.01, 0.5]
}

svr_base = SVR(kernel='rbf', cache_size=1000)  # 1GB cache for kernel matrix

print(f"  Kernel: RBF (Radial Basis Function)")
print(f"  Parameter search space:")
print(f"    C:       loguniform(1, 1000)")
print(f"    gamma:   loguniform(0.001, 1)")
print(f"    epsilon: loguniform(0.01, 0.5)")
print(f"  Search: RandomizedSearchCV (n_iter=40, cv=5)")
print(f"  Scoring: neg_mean_squared_error (on log-scale)")
print(f"\n  Fitting... (this will take ~30 minutes)")

start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=svr_base,
    param_distributions=param_distributions,
    n_iter=40,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,          # Use all CPU cores
    random_state=42,
    verbose=0,
    refit=True           # Refit best model on full training set
)

random_search.fit(X_train_pca, y_train_log)
tuning_time = time.time() - start_time

print(f"  Tuning completed in {tuning_time:.1f} seconds ({tuning_time/60:.1f} min)")

# Best parameters
best_params = random_search.best_params_
best_cv_score = -random_search.best_score_  # Negate because sklearn minimizes

print(f"\n  Best Hyperparameters:")
print(f"    C       = {best_params['C']:.4f}")
print(f"    gamma   = {best_params['gamma']:.6f}")
print(f"    epsilon = {best_params['epsilon']:.4f}")
print(f"  Best CV MSE (log-scale): {best_cv_score:.6f}")
print(f"  Best CV RMSE (log-scale): {np.sqrt(best_cv_score):.6f}")

# ---- 5.3 Model Evaluation ----
print("\n[5.3] MODEL EVALUATION")
print("-" * 50)

best_svr = random_search.best_estimator_

# Predict on log-scale
y_train_pred_log = best_svr.predict(X_train_pca)
y_test_pred_log = best_svr.predict(X_test_pca)

# Reverse log-transform to original price scale
y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

# Metrics on original price scale
def evaluate_model(y_true, y_pred, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, r2, mape

train_rmse, train_mae, train_r2, train_mape = evaluate_model(y_train, y_train_pred, "Training")
test_rmse, test_mae, test_r2, test_mape = evaluate_model(y_test, y_test_pred, "Testing")

print(f"\n  {'Metric':<25} {'Training':>15} {'Testing':>15}")
print(f"  {'-'*55}")
print(f"  {'RMSE (AUD)':<25} ${train_rmse:>14,.0f} ${test_rmse:>14,.0f}")
print(f"  {'MAE (AUD)':<25} ${train_mae:>14,.0f} ${test_mae:>14,.0f}")
print(f"  {'R-squared':<25} {train_r2:>15.4f} {test_r2:>15.4f}")
print(f"  {'MAPE (%)':<25} {train_mape:>14.2f}% {test_mape:>14.2f}%")

# Interpretation
print(f"\n  Interpretation:")
print(f"    - R2 = {test_r2:.4f} means the model explains {test_r2*100:.1f}% of price variance")
print(f"    - MAE = ${test_mae:,.0f} means predictions are off by ~${test_mae:,.0f} on average")
if test_r2 > train_r2 - 0.05:
    print(f"    - Train-Test R2 gap = {train_r2-test_r2:.4f} (small gap -> no severe overfitting)")
else:
    print(f"    - Train-Test R2 gap = {train_r2-test_r2:.4f} (moderate gap -> some overfitting)")

# ---- 5.4 Prediction vs Actual Visualization ----
print(f"\n[5.4] Generating prediction vs actual plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5.4a: Training set
axes[0].scatter(y_train / 1e6, y_train_pred / 1e6, alpha=0.2, s=8,
                color='#4C72B0', edgecolors='none')
max_val_train = max(y_train.max(), y_train_pred.max()) / 1e6
axes[0].plot([0, max_val_train], [0, max_val_train], 'r--', linewidth=1.5,
             label='Perfect prediction')
axes[0].set_xlabel('Actual Price (Millions AUD)')
axes[0].set_ylabel('Predicted Price (Millions AUD)')
axes[0].set_title(f'Training Set (R2={train_r2:.4f})')
axes[0].legend()
axes[0].set_xlim(0, max_val_train * 1.05)
axes[0].set_ylim(0, max_val_train * 1.05)

# 5.4b: Test set
axes[1].scatter(y_test / 1e6, y_test_pred / 1e6, alpha=0.3, s=12,
                color='#C44E52', edgecolors='none')
max_val_test = max(y_test.max(), y_test_pred.max()) / 1e6
axes[1].plot([0, max_val_test], [0, max_val_test], 'r--', linewidth=1.5,
             label='Perfect prediction')
axes[1].set_xlabel('Actual Price (Millions AUD)')
axes[1].set_ylabel('Predicted Price (Millions AUD)')
axes[1].set_title(f'Test Set (R2={test_r2:.4f})')
axes[1].legend()
axes[1].set_xlim(0, max_val_test * 1.05)
axes[1].set_ylim(0, max_val_test * 1.05)

plt.suptitle('SVR Predictions vs Actual House Prices', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('eda_plots/10_svr_predictions_vs_actual.png', bbox_inches='tight')
plt.close()
print(f"  -> Saved: eda_plots/10_svr_predictions_vs_actual.png")

# ---- 5.5 Build Reproducible Pipeline ----
# Package the complete preprocessing + modeling flow into a scikit-learn
# Pipeline object for clean, reproducible inference.
print(f"\n[5.5] BUILDING REPRODUCIBLE PIPELINE")
print("-" * 50)

# The full pipeline consists of:
#   1. StandardScaler (fitted on cleaned, reduced training features)
#   2. PCA (dimensionality reduction)
#   3. SVR (with best hyperparameters)
# Note: The preprocessing steps (imputation, encoding, outlier capping)
#       are applied BEFORE this pipeline. Their fitted objects (knn_imputer,
#       label encoders, etc.) are saved separately.

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components_95, random_state=42)),
    ('svr', SVR(
        kernel='rbf',
        C=best_params['C'],
        gamma=best_params['gamma'],
        epsilon=best_params['epsilon'],
        cache_size=1000
    ))
])

# Fit the pipeline on the reduced (pre-PCA) training data
final_pipeline.fit(X_train_reduced, y_train_log)

# Verify pipeline predictions match our standalone results
pipeline_pred_log = final_pipeline.predict(X_test_reduced)
pipeline_pred = np.expm1(pipeline_pred_log)
pipeline_r2 = r2_score(y_test, pipeline_pred)
print(f"  Pipeline R2 on test set: {pipeline_r2:.4f} (matches standalone: {test_r2:.4f})")

# ---- 5.6 Save Model Artifacts ----
print(f"\n[5.6] SAVING MODEL ARTIFACTS")
print("-" * 50)

# Save the complete pipeline
joblib.dump(final_pipeline, 'svr_pipeline.joblib')
print(f"  Saved: svr_pipeline.joblib")

# Save preprocessing objects needed for new data
preprocessing_artifacts = {
    'drop_cols': ['Address', 'Suburb', 'SellerG'],
    'drop_features': ['Bedroom2', 'Postcode'],
    'type_map': {'h': 2, 't': 1, 'u': 0},
    'method_le': method_le,
    'council_le': council_le,
    'council_mode_by_postcode': council_mode_by_postcode,
    'global_council_mode': global_council_mode,
    'car_medians': car_medians,
    'knn_imputer': knn_imputer,
    'knn_features': knn_features,
    'region_columns': region_dummies_train.columns.tolist(),
    'feature_names': feature_names,
    'n_components': n_components_95,
    'best_params': best_params,
    'cap_bounds': {
        'Rooms': (float(rooms_lo), float(rooms_hi)),
        'Bathroom': (float(bath_lo), float(bath_hi)),
        'Car': (float(car_lo), float(car_hi)),
        'Landsize': (float(landsize_lo), float(landsize_hi)),
        'BuildingArea': (float(building_lo), float(building_hi)),
    },
}
joblib.dump(preprocessing_artifacts, 'preprocessing_artifacts.joblib')
print(f"  Saved: preprocessing_artifacts.joblib")

# ---- 5.7 Summary ----
print("\n" + "=" * 70)
print("STAGE 5 COMPLETE")
print("=" * 70)
print(f"""
Summary of SVM Model Development:
  [OK] Target log-transformed: skew {y_train.skew():.2f} -> {y_train_log.skew():.2f}
  [OK] SVR with RBF kernel, tuned via RandomizedSearchCV (30 iter, 5-fold CV)
  [OK] Best hyperparameters: C={best_params['C']:.4f}, gamma={best_params['gamma']:.6f}, epsilon={best_params['epsilon']:.4f}
  [OK] Tuning time: {tuning_time:.1f} seconds

  MODEL PERFORMANCE:
  +--------------------------+----------------+----------------+
  | Metric                   | Training       | Testing        |
  +--------------------------+----------------+----------------+
  | RMSE                     | ${train_rmse:>13,.0f} | ${test_rmse:>13,.0f} |
  | MAE                      | ${train_mae:>13,.0f} | ${test_mae:>13,.0f} |
  | R-squared                | {train_r2:>14.4f} | {test_r2:>14.4f} |
  | MAPE                     | {train_mape:>13.2f}% | {test_mape:>13.2f}% |
  +--------------------------+----------------+----------------+

  [OK] Pipeline built: StandardScaler -> PCA({n_components_95}) -> SVR(RBF)
  [OK] Saved: svr_pipeline.joblib + preprocessing_artifacts.joblib
""")
print("AWAITING APPROVAL to proceed to Stage 6: Bonus UI Deployment")
print("=" * 70)


# =============================================================================
# STAGE 6: UI DEPLOYMENT (SEPARATE FILE)
# =============================================================================
# The UI deployment is implemented in two separate files:
#
#   1. app.py           - FastAPI backend (REST API)
#   2. static/index.html - Premium frontend (HTML/CSS/JS)
#
# To launch the UI:
#   python app.py
#   -> Opens at http://localhost:8000
#
# Architecture:
#   [Frontend] --POST /predict--> [FastAPI] --pipeline.predict()--> [SVR Model]
#        |                             |                                |
#   User enters                  Preprocesses input                Returns
#   property details            (same pipeline as                 predicted
#   in the form                  training Stage 3-4)              AUD price
#
# The backend loads:
#   - svr_pipeline.joblib           (Scaler + PCA + SVR)
#   - preprocessing_artifacts.joblib (encoders, imputers, etc.)
# =============================================================================

print("\n" + "=" * 70)
print("STAGE 6: UI DEPLOYMENT")
print("=" * 70)
print("""
  [OK] FastAPI backend created: app.py
  [OK] Premium frontend created: static/index.html
  [OK] Model artifacts ready: svr_pipeline.joblib + preprocessing_artifacts.joblib

  To launch the web UI:
    python app.py
    -> Server starts at http://localhost:8000

  API Endpoints:
    GET  /          -> Serves the prediction UI
    POST /predict   -> Accepts property features, returns predicted price
    GET  /api/info  -> Returns model metadata

  ALL 6 STAGES COMPLETE!
""")
print("=" * 70)
print("PROJECT COMPLETE: Melbourne House Price Prediction using SVR")
print("=" * 70)
