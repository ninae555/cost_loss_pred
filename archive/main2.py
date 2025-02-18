#%%
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import TweedieRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tensorflow.keras.losses import MeanSquaredError

# **🔹 Model Paths**
MODEL_DIR = "saved_models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
NN_MODEL_PATH = os.path.join(MODEL_DIR, "nn_model.h5")
GLM_MODEL_PATH = os.path.join(MODEL_DIR, "glm_model.pkl")
CAT_MODEL_PATH = os.path.join(MODEL_DIR, "cat_model.pkl")

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# **🔹 Load Data**
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# **🔹 Feature Engineering**
train_df["DateTimeOfAccident"] = pd.to_datetime(train_df["DateTimeOfAccident"])
train_df["DateReported"] = pd.to_datetime(train_df["DateReported"])
train_df["ReportDelay"] = (train_df["DateReported"] - train_df["DateTimeOfAccident"]).dt.days

test_df["DateTimeOfAccident"] = pd.to_datetime(test_df["DateTimeOfAccident"])
test_df["DateReported"] = pd.to_datetime(test_df["DateReported"])
test_df["ReportDelay"] = (test_df["DateReported"] - test_df["DateTimeOfAccident"]).dt.days

# **🔹 Compute Season Feature**
def get_season(date):
    month = date.month
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Fall"

train_df["Season"] = train_df["DateTimeOfAccident"].apply(get_season)
test_df["Season"] = test_df["DateTimeOfAccident"].apply(get_season)

# **🔹 Convert Categorical Features to 'category' dtype**
categorical_features = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'Season']
for col in categorical_features:
    train_df[col] = train_df[col].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")
    test_df[col] = test_df[col].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")

# **🔹 NLP Processing for ClaimDescription**
vectorizer = TfidfVectorizer(max_features=100)
train_text_features = vectorizer.fit_transform(train_df["ClaimDescription"].astype(str)).toarray()
test_text_features = vectorizer.transform(test_df["ClaimDescription"].astype(str)).toarray()

train_text_df = pd.DataFrame(train_text_features, columns=[f"text_feat_{i}" for i in range(100)])
test_text_df = pd.DataFrame(test_text_features, columns=[f"text_feat_{i}" for i in range(100)])

train_df = pd.concat([train_df.reset_index(drop=True), train_text_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_text_df], axis=1)

# **📌 Define Features & Target**
numeric_features = ['Age', 'WeeklyWages', 'DependentChildren', 'DependentsOther', 
                    'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'ReportDelay', 'InitialIncurredCalimsCost']
features = numeric_features + categorical_features + [f"text_feat_{i}" for i in range(100)]

X_train = train_df[features]
y_train = train_df["UltimateIncurredClaimCost"]
X_test = test_df[features]

# **🔹 Preprocessing Pipeline**
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  
])

#%%
# **🔹 Train-Validation Split**
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# **🔹 Train XGBoost Model with Validation**
if os.path.exists(XGB_MODEL_PATH):
    with open(XGB_MODEL_PATH, "rb") as file:
        xgb_model = pickle.load(file)
    print("✅ XGBoost Model Loaded!")
else:
    # Transform training data
    X_train_transformed = preprocessor.fit_transform(X_train_final)
    X_val_transformed = preprocessor.transform(X_val)

    # ✅ FIX: Move `eval_metric="rmse"` to constructor instead of `.fit()`
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric="rmse")

    xgb_model.fit(
        X_train_transformed, y_train_final, 
        eval_set=[(X_train_transformed, y_train_final), (X_val_transformed, y_val)],
        verbose=50,
        early_stopping_rounds=20
    )

    # Save model
    with open(XGB_MODEL_PATH, "wb") as file:
        pickle.dump(xgb_model, file)
    print("✅ XGBoost Model Trained & Saved!")

# **🔹 Train CatBoost Model with Validation**
cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]

if os.path.exists(CAT_MODEL_PATH):
    with open(CAT_MODEL_PATH, "rb") as file:
        cat_model = pickle.load(file)
    print("✅ CatBoost Model Loaded!")
else:
    cat_model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=50, eval_metric='RMSE')
    cat_model.fit(X_train_final, y_train_final, eval_set=[(X_val, y_val)], cat_features=cat_feature_indices)
    
    with open(CAT_MODEL_PATH, "wb") as file:
        pickle.dump(cat_model, file)
    print("✅ CatBoost Model Trained & Saved!")

# **🔹 Compute & Plot Loss for Each Model**
plt.figure(figsize=(12, 6))

# **🔹 XGBoost Loss Plot**
if hasattr(xgb_model, "evals_result"):
    xgb_evals = xgb_model.evals_result()
    if "validation_0" in xgb_evals:
        plt.plot(xgb_evals["validation_0"]["rmse"], label="XGBoost Train RMSE")
    if "validation_1" in xgb_evals:
        plt.plot(xgb_evals["validation_1"]["rmse"], label="XGBoost Validation RMSE")

# **🔹 CatBoost Loss Plot**
if hasattr(cat_model, "get_evals_result"):
    cat_evals = cat_model.get_evals_result()
    plt.plot(cat_evals["learn"]["RMSE"], label="CatBoost Train RMSE")
    if "validation" in cat_evals:
        plt.plot(cat_evals["validation"]["RMSE"], label="CatBoost Validation RMSE")

plt.xlabel("Epochs / Iterations")
plt.ylabel("Loss (RMSE)")
plt.title("Training vs. Validation Loss for Each Model")
plt.legend()
plt.grid()
plt.show()

# **🔹 Predict & Create Ensemble Model**
y_pred_xgb = xgb_model.predict(preprocessor.transform(X_test))
y_pred_cat = cat_model.predict(X_test)
y_pred_ensemble = (y_pred_xgb + y_pred_cat) / 2

test_df["PredictedClaimCost_Ensemble_Advanced"] = y_pred_ensemble
test_df[["ClaimNumber", "PredictedClaimCost_Ensemble_Advanced"]].to_csv("ensemble_predictions_advanced.csv", index=False)

print("🚀 Model Training Complete! Checkpointing Enabled.")


#%%

# **🔹 Model Monitoring & Validation**
train_df["Prediction_Error"] = train_df["UltimateIncurredClaimCost"] - xgb_pipeline.predict(X_train)

plt.figure(figsize=(10,5))
train_df["Prediction_Error"].rolling(50).mean().plot()
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel("Claims Processed")
plt.ylabel("Rolling Mean Prediction Error")
plt.title("Model Residual Trend Over Time (Drift Detection)")
plt.show()

# **🔹 Feature Importance Stability**
perm_importance = permutation_importance(xgb_pipeline, X_train, y_train, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10,5))
plt.barh(np.array(features)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Feature Importance (Permutation)")
plt.ylabel("Feature Name")
plt.title("Feature Importance Stability Over Time")
plt.show()

# %%
