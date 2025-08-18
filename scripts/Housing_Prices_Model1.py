# =========================
# Cell 0: Imports & Setup
# =========================
import os
import numpy as np
import pandas as pd

from scipy.stats import skew
from pathlib import Path

DATA_DIR = Path(r"C:\Users\rcoul\Downloads\Housing_Prices_Model\Data\Raw")
OUT_DIR = Path(r"C:\Users\rcoul\Downloads\Housing_Prices_Model\Data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 120)

# =========================
# Cell 1: Load raw data
# =========================
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(train.shape, test.shape)
train.head(3)

# =========================
# Cell 2: Combine for uniform cleaning
# (keep a flag to split later)
# =========================
train['__is_train'] = 1
test['__is_train'] = 0
full = pd.concat([train, test], axis=0, ignore_index=True)
print(full.shape)

# =========================
# Cell 3: Quick missingness snapshot
# =========================
miss = full.isna().mean().sort_values(ascending=False)
miss[miss > 0].head(20)

# =========================
# Cell 4: Fix data types (categorical vs ordinal vs numeric)
# - Many "numbers" are actually categories (e.g., MSSubClass, MoSold)
# =========================
# treat as categorical strings
cat_as_num = ["MSSubClass", "MoSold", "YrSold"]
full[cat_as_num] = full[cat_as_num].astype(str)

# Ordinal quality maps (Ex>Gd>TA>Fa>Po). Include None->0 for convenience.
qual_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}

for col in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
            "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
    full[col] = full[col].map(qual_map).astype("Int64")

# Basement exposure & finish types (custom ordinal encodings)
bsmt_exp_map = {"Gd":4, "Av":3, "Mn":2, "No":1, np.nan:0}
for col in ["BsmtExposure"]:
    full[col] = full[col].map(bsmt_exp_map).astype("Int64")

bsmt_fin_map = {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1, np.nan:0}
for col in ["BsmtFinType1","BsmtFinType2"]:
    full[col] = full[col].map(bsmt_fin_map).astype("Int64")

# NA means "None" (feature absent)
none_fill = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
             "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
             "PoolQC","Fence","MiscFeature","MasVnrType"]
for col in none_fill:
    if col in full.columns:
        full[col] = full[col].fillna("None")

# Garage numerics: NA -> 0
for col in ["GarageYrBlt","GarageArea","GarageCars"]:
    if col in full.columns:
        full[col] = full[col].fillna(0)

# Masonry veneer area: NA -> 0
if "MasVnrArea" in full.columns:
    full["MasVnrArea"] = full["MasVnrArea"].fillna(0)

# LotFrontage by neighborhood median
if "LotFrontage" in full.columns and "Neighborhood" in full.columns:
    full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )

# Functional: NA -> Typ
if "Functional" in full.columns:
    full["Functional"] = full["Functional"].fillna("Typ")

# Mode fills for a few mostly-complete categoricals
for col in ["MSZoning","Electrical","KitchenQual","Exterior1st","Exterior2nd","SaleType","Utilities"]:
    if col in full.columns and full[col].isna().any():
        full[col] = full[col].fillna(full[col].mode()[0])

# Totals
full["TotalSF"] = full.get("1stFlrSF",0) + full.get("2ndFlrSF",0) + full.get("TotalBsmtSF",0)
full["TotalBath"] = (
    full.get("FullBath",0) + 0.5*full.get("HalfBath",0) +
    full.get("BsmtFullBath",0) + 0.5*full.get("BsmtHalfBath",0)
)

# Flags
full["HasPool"] = (full.get("PoolArea",0) > 0).astype(int)
full["HasGarage"] = (full.get("GarageArea",0) > 0).astype(int)
full["HasFireplace"] = (full.get("Fireplaces",0) > 0).astype(int)
full["HasBasement"] = (full.get("TotalBsmtSF",0) > 0).astype(int)

# Ages
if {"YrSold","YearBuilt"}.issubset(full.columns):
    full["HouseAge"] = full["YrSold"].astype(int) - full["YearBuilt"].astype(int)
if {"YrSold","YearRemodAdd"}.issubset(full.columns):
    full["RemodAge"] = full["YrSold"].astype(int) - full["YearRemodAdd"].astype(int)
if "GarageYrBlt" in full.columns and "YrSold" in full.columns:
    full["GarageAge"] = np.where(full["GarageYrBlt"]>0,
                                 full["YrSold"].astype(int)-full["GarageYrBlt"].astype(int),
                                 -1)
    
# =========================
# Cell 6: Context-aware imputations
# =========================
# LotFrontage: impute by Neighborhood median
if "LotFrontage" in full.columns:
    full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )

# Functional: NA -> Typ
if "Functional" in full.columns:
    full["Functional"] = full["Functional"].fillna("Typ")

# Mode fills for a few mostly-complete categoricals
for col in ["MSZoning","Electrical","KitchenQual","Exterior1st","Exterior2nd","SaleType","Utilities"]:
    if col in full.columns and full[col].isna().any():
        full[col] = full[col].fillna(full[col].mode()[0])

# =========================
# Cell 7: Feature engineering (tidy + useful combos)
# =========================
# Total square footage including basement
full["TotalSF"] = full.get("1stFlrSF",0) + full.get("2ndFlrSF",0) + full.get("TotalBsmtSF",0)

# Total baths (weighted half baths)
full["TotalBath"] = (
    full.get("FullBath",0) + 0.5*full.get("HalfBath",0) +
    full.get("BsmtFullBath",0) + 0.5*full.get("BsmtHalfBath",0)
)

# Has flags
full["HasPool"] = (full.get("PoolArea",0) > 0).astype(int)
full["HasGarage"] = (full.get("GarageArea",0) > 0).astype(int)
full["HasFireplace"] = (full.get("Fireplaces",0) > 0).astype(int)
full["HasBasement"] = (full.get("TotalBsmtSF",0) > 0).astype(int)

# Age features
full["HouseAge"] = full["YrSold"].astype(int) - full["YearBuilt"].astype(int)
full["RemodAge"] = full["YrSold"].astype(int) - full["YearRemodAdd"].astype(int)
full["GarageAge"] = np.where(full["GarageYrBlt"]>0, full["YrSold"].astype(int)-full["GarageYrBlt"].astype(int), -1)

# =========================
# Cell 8: Rare category consolidation
# - Collapse very small levels to "Other" to reduce one-hot sparsity
# =========================
def collapse_rare_categories(series, min_count=10, other_label="Other"):
    vc = series.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return series.where(~series.isin(rare), other_label)

cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    full[col] = collapse_rare_categories(full[col], min_count=10)

# =========================
# Cell 9: Drop nearly-constant/unhelpful cols (optional)
# =========================
to_drop = []
if "Utilities" in full.columns and full["Utilities"].nunique() == 1:
    to_drop.append("Utilities") # almost always 'AllPub'
# (Optionally drop highly collinear duplicates later for linear models)
full = full.drop(columns=to_drop)

# =========================
# Cell 10: Handle obvious outliers (train only)
# - Common heuristic: remove huge GrLivArea with suspiciously low SalePrice
# =========================
# We'll mark them now and actually remove after we split back to train.
full["_outlier_flag"] = 0
if "GrLivArea" in full.columns and "SalePrice" in full.columns:
    mask_out = (full["__is_train"]==1) & (full["GrLivArea"]>4000) & (full["SalePrice"]<300000)
    full.loc[mask_out, "_outlier_flag"] = 1
full["_outlier_flag"].value_counts()

# =========================
# Cell 11: Transform skewed numeric features (log1p)
# - Do NOT transform target here; that belongs in modeling.
# =========================
numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()
# exclude target and helper columns
exclude = {"SalePrice","Id","__is_train","_outlier_flag","GarageYrBlt"}
num_feats = [c for c in numeric_cols if c not in exclude]

skews = full[num_feats].apply(lambda s: skew(s.dropna())).sort_values(ascending=False)
skewed = skews[skews > 0.75].index.tolist()

# log1p transform skewed positive features (avoid negatives)
for col in skewed:
    # shift if needed to ensure positivity
    min_val = full[col].min()
    if pd.notna(min_val) and min_val <= 0:
        full[col] = full[col] - min_val + 1
    full[col] = np.log1p(full[col])
    
skews.head(10), len(skewed)

# =========================
# Cell 12: One-hot encode remaining categoricals
# =========================
cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=False)
full_encoded.shape

# =========================
# Cell 13: Split back to train/test, drop outliers from train
# =========================
train_clean = full_encoded[full_encoded["__is_train"]==1].copy()
test_clean = full_encoded[full_encoded["__is_train"]==0].copy()

# drop helper cols
for c in ["__is_train"]:
    if c in train_clean.columns: train_clean = train_clean.drop(columns=[c])
    if c in test_clean.columns: test_clean = test_clean.drop(columns=[c])

# remove marked outliers from training set
train_clean = train_clean[train_clean["_outlier_flag"]==0].copy()
train_clean = train_clean.drop(columns=["_outlier_flag"])
if "_outlier_flag" in test_clean.columns:
    test_clean = test_clean.drop(columns=["_outlier_flag"])

print(train_clean.shape, test_clean.shape)

# =========================
# Cell 14: Align columns (ensure test has same cols)
# =========================
# Align columns so X_train and X_test match (excluding SalePrice)
y = train_clean["SalePrice"].copy()
X = train_clean.drop(columns=["SalePrice"])
X_test = test_clean.drop(columns=["SalePrice"]) if "SalePrice" in test_clean.columns else test_clean.copy()

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
X.shape, X_test.shape

# =========================
# Cell 15: Save cleaned outputs for modeling
# =========================
X.to_csv(OUT_DIR / "train_features_clean.csv", index=False)
y.to_csv(OUT_DIR / "train_target.csv", index=False)
X_test.to_csv(OUT_DIR / "test_features_clean.csv", index=False)

print("Saved to /kaggle/working:")
list(OUT_DIR.iterdir())

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

IN_KAGGLE = Path("/kaggle/working").exists()

if IN_KAGGLE:
    PROC_DIR = Path("/kaggle/working")  # where you saved the cleaned files in Kaggle
    RAW_DIR  = Path("/kaggle/input/house-prices-advanced-regression-techniques")
else:
    BASE    = Path(r"C:\Users\rcoul\Downloads\Housing_Prices_Model\Data")
    PROC_DIR = BASE / "Processed"
    RAW_DIR  = BASE / "Raw"

print("PROC_DIR:", PROC_DIR)
print("RAW_DIR :", RAW_DIR)

# ---- load processed ----
X_train = pd.read_csv(PROC_DIR / "train_features_clean.csv")
y_train = pd.read_csv(PROC_DIR / "train_target.csv").squeeze("columns")
X_test = pd.read_csv(PROC_DIR / "test_features_clean.csv")

# ---- handle any remaining NaNs ----
imputer = SimpleImputer(strategy="median")
X_train_i = imputer.fit_transform(X_train)
X_test_i = imputer.transform(X_test)

# ---- baseline model + CV ----
model = RandomForestRegressor(n_estimators=200, random_state=42)
cv = cross_val_score(model, X_train_i, y_train, cv=5, scoring="neg_root_mean_squared_error")
print("CV RMSE:", -cv.mean())

# ---- fit & predict ----
model.fit(X_train_i, y_train)
preds = model.predict(X_test_i)

# ---- build submission from official sample ----
sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
sample["SalePrice"] = preds

# save submission in the right place for each env
sub_path = PROC_DIR / "submission.csv"
sample.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)