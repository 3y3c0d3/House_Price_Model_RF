##🌲 House Prices Prediction with Random Forest

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-ScikitLearn%20%7C%20Pandas%20%7C%20NumPy-orange.svg)
![Competition](https://img.shields.io/badge/Kaggle-House%20Prices%20Competition-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project uses a **Random Forest Regressor** to predict house prices for the Kaggle competition **House Prices - Advanced Regression Techniques**.  
It demonstrates how to build an **end-to-end machine learning workflow** with data cleaning, feature engineering, and evaluation.  

---

## 📂 Project Structure

house-prices-random-forest/ │── data/ │ ├── train.csv # Raw training data (from Kaggle) │ ├── test.csv # Raw test data (from Kaggle) │ ├── train_features_clean.csv # Cleaned training features │ ├── test_features_clean.csv # Cleaned test features │ └── train_target.csv # Target variable (SalePrice) │ │── notebooks/ │ └── housing_prices_randomforest.ipynb # Kaggle Notebook / Jupyter Notebook │ │── scripts/ │ └── Housing_Prices_RF_Model.py # Python script version of pipeline │  │── README.md # Project documentation

---

## ⚙️ Workflow

1. **Data Cleaning**  
   - Filled missing values (median for numeric, mode for categorical).  
   - Encoded categorical features with OneHotEncoder.  

2. **Feature Engineering**  
   - Separated categorical and numeric features.  
   - Built preprocessing pipeline with `ColumnTransformer`.  

3. **Modeling**  
   - Random Forest Regressor (`RandomForestRegressor`) with:  
     - `n_estimators=200`  
     - `max_depth=None` (allow full growth)  
     - `random_state=42`  

4. **Evaluation**  
   - Metric: Root Mean Squared Log Error (RMSLE).  
   - Cross-Validation RMSE achieved: **~0.138** (baseline improvement).  
   - Kaggle Private Leaderboard Score: **0.14x**.  

---

## 📊 Results

- ✅ Strong baseline model without heavy tuning.  
- ✅ Showed improvement over Linear Regression baseline.  
- ✅ Served as a foundation before advancing to **XGBoost**.  

---

## 🚀 How to Reproduce

Clone this repository and run:

```bash
git clone https://github.com/<your-username>/house-prices-random-forest.git
cd house-prices-random-forest
pip install -r requirements.txt
python scripts/Housing_Prices_RF_Model.py

Or open the Kaggle Notebook directly and rerun all cells.


```

## 🛠️ Tech Stack

Python (3.9+)

Pandas / NumPy – data manipulation

Scikit-Learn – Random Forest, pipelines, preprocessing

Matplotlib / Seaborn – visualizations



---

## 📌 Acknowledgements

Dataset: Kaggle House Prices Competition

Inspired by top Kaggle kernels and community notebooks.



---

## ✨ Author

👤 Ryan Coulter
🔗 GitHub | Kaggle | LinkedIn
