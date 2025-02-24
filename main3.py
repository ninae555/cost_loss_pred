# %%
# Import Libraries 
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
from xgboost import XGBRegressor, callback
from sklearn.linear_model import TweedieRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from lightgbm import LGBMRegressor, log_evaluation
from lightgbm.callback import early_stopping
from tensorflow.keras.regularizers import l2
from scipy.stats import ttest_ind, f_oneway
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.model_selection import train_test_split

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

print(train_df.head())

#%%

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

print(train_df.head())

#%%
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

# **🔹 Train-Validation Split**
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)


# **🔹 Train Generalized Linear Model (GLM)**
glm_model = TweedieRegressor(power=1.5, alpha=0.5, max_iter=1000)
glm_model.fit(X_train_final, y_train_final)
y_pred_glm = glm_model.predict(X_test_scaled)

# **🔹 Train Neural Network (With Early Stopping & Weight Decay)**
nn_model = Sequential([
     Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train_scaled.shape[1],)),
     Dropout(0.3),
     Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
     Dropout(0.2),
     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
     Dense(1)
 ])

nn_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("✅ Ensemble Model Training Complete & Predictions Saved!")
#%%
import matplotlib.pyplot as plt

# **🔹 Separate Loss Plots for Each Model**
fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # 3 rows, 1 column for separate plots

 # **🔹 Neural Network Loss Plot**
axes[0].plot(history.history['loss'], label='NN Train Loss')
axes[0].plot(history.history['val_loss'], label='NN Validation Loss')
axes[0].set_title("Neural Network Training vs. Validation Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss (MSE)")
axes[0].legend()
axes[0].grid()

# **🔹 XGBoost Loss Plot**
if hasattr(xgb_model, "evals_result"):
    xgb_evals = xgb_model.evals_result()
    if "validation_0" in xgb_evals:
        axes[1].plot(xgb_evals["validation_0"]["rmse"], label="XGBoost Train RMSE")
    if "validation_1" in xgb_evals:
        axes[1].plot(xgb_evals["validation_1"]["rmse"], label="XGBoost Validation RMSE")

axes[1].set_title("XGBoost Training vs. Validation Loss")
axes[1].set_xlabel("Epochs / Iterations")
axes[1].set_ylabel("Loss (RMSE)")
axes[1].legend()
axes[1].grid()

# **🔹 CatBoost Loss Plot**
if hasattr(cat_model, "get_evals_result"):
    cat_evals = cat_model.get_evals_result()
    axes[2].plot(cat_evals["learn"]["RMSE"], label="CatBoost Train RMSE")
    if "validation" in cat_evals:
        axes[2].plot(cat_evals["validation"]["RMSE"], label="CatBoost Validation RMSE")

axes[2].set_title("CatBoost Training vs. Validation Loss")
axes[2].set_xlabel("Epochs / Iterations")
axes[2].set_ylabel("Loss (RMSE)")
axes[2].legend()
axes[2].grid()

# **🔹 Adjust layout for clarity**
plt.tight_layout()
plt.show()


# %%
# **🔹 Evaluate Performance**
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# **🔹 Evaluate Each Model**

evaluate_model(y_val, glm_model.predict(X_val), "GLM")
evaluate_model(y_val, nn_model.predict(X_val).flatten(), "Neural Network")
#evaluate_model(y_val, y_pred_ensemble, "Ensemble Model")

# **🔹 Statistical Analysis**
married_costs = train_df[train_df["MaritalStatus"] == "M"]["UltimateIncurredClaimCost"]
single_costs = train_df[train_df["MaritalStatus"] == "S"]["UltimateIncurredClaimCost"]
t_stat, p_val = ttest_ind(married_costs, single_costs, equal_var=False)
print(f"T-Test (Married vs. Single) - p-value: {p_val:.4f}")

ft_claims = train_df[train_df["PartTimeFullTime"] == "F"]["UltimateIncurredClaimCost"]
pt_claims = train_df[train_df["PartTimeFullTime"] == "P"]["UltimateIncurredClaimCost"]
f_stat, p_val = f_oneway(ft_claims, pt_claims)
print(f"ANOVA (Part-Time vs. Full-Time) - p-value: {p_val:.4f}")

# **🔹 Residual Analysis for Drift Detection**
train_df["Prediction_Error"] = train_df["UltimateIncurredClaimCost"] - xgb_model.predict(X_train_scaled)

plt.figure(figsize=(10, 5))
train_df["Prediction_Error"].rolling(50).mean().plot()
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel("Claims Processed")
plt.ylabel("Rolling Mean Prediction Error")
plt.title("Model Residual Trend Over Time (Drift Detection)")
plt.show()

# **🔹 Feature Importance Stability**
perm_importance = permutation_importance(xgb_model, X_train_scaled, y_train, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 5))
plt.barh(np.array(features)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Feature Importance (Permutation)")
plt.ylabel("Feature Name")
plt.title("Feature Importance Stability Over Time")
plt.show()
# %%
