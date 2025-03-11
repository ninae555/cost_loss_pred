# %%
# Import Libraries 
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
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, f_oneway
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# **🔹 Load Data**
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


# **🔹 Train XGBoost Model**
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

# **🔹 Train Neural Network**
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, verbose=1,
    callbacks=[early_stop]
)

y_pred_nn = nn_model.predict(X_test_scaled).flatten()

# **🔹 Ensemble Model (Average of XGBoost & NN Predictions)**
y_pred_ensemble = (y_pred_xgb + y_pred_nn) / 2
test_df["PredictedClaimCost_Ensemble"] = y_pred_ensemble
test_df[["ClaimNumber", "PredictedClaimCost_Ensemble"]].to_csv("ensemble_predictions.csv", index=False)

print("✅ Ensemble Model Training Complete & Predictions Saved!")

# **🔹 Evaluate Performance**
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

evaluate_model(y_train, xgb_pipeline.predict(X_train), "XGBoost")
evaluate_model(y_train, nn_model.predict(X_train_scaled).flatten(), "Neural Network")
evaluate_model(y_train, (xgb_pipeline.predict(X_train) + nn_model.predict(X_train_scaled).flatten()) / 2, "Ensemble Model")

# **🔹 Fraud Detection**
train_df["ClaimCostRatio"] = train_df["UltimateIncurredClaimCost"] / (train_df["InitialIncurredCalimsCost"] + 1)

plt.figure(figsize=(10, 5))
sns.histplot(train_df["ClaimCostRatio"], bins=50, kde=True)
plt.axvline(train_df["ClaimCostRatio"].mean(), color='r', linestyle='dashed', linewidth=2)
plt.xlabel("ClaimCostRatio")
plt.ylabel("Count")
plt.title("Claim Cost Ratio Distribution (Potential Fraud Detection)")
plt.show()

fraud_threshold = train_df["ClaimCostRatio"].quantile(0.99)
potential_fraud_cases = train_df[train_df["ClaimCostRatio"] > fraud_threshold]
print(f"\nPotential Fraud Cases: {len(potential_fraud_cases)}")

# **🔹 Statistical Analysis**
married_costs = train_df[train_df["MaritalStatus"] == "M"]["UltimateIncurredClaimCost"]
single_costs = train_df[train_df["MaritalStatus"] == "S"]["UltimateIncurredClaimCost"]
t_stat, p_val = ttest_ind(married_costs, single_costs, equal_var=False)
print(f"T-Test (Married vs. Single) - p-value: {p_val:.4f}")

ft_claims = train_df[train_df["PartTimeFullTime"] == "F"]["UltimateIncurredClaimCost"]
pt_claims = train_df[train_df["PartTimeFullTime"] == "P"]["UltimateIncurredClaimCost"]
f_stat, p_val = f_oneway(ft_claims, pt_claims)
print(f"ANOVA (Part-Time vs. Full-Time) - p-value: {p_val:.4f}")

# **🔹 Plot Neural Network Training Loss**
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Neural Network Training vs. Validation Loss')
plt.legend()
plt.show()

# %%
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
from sklearn.inspection import permutation_importance

# Compute permutation importance for XGBoost
perm_importance = permutation_importance(
    xgb_pipeline,  # Trained pipeline
    X_train, y_train,  # Train data & labels
    n_repeats=10,  # Number of shuffles
    random_state=42,
    scoring='neg_mean_squared_error'  # Use MSE as the metric
)

# Sort feature importance in descending order
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

# **🔹 Plot Feature Importance**
plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Feature Importance (Permutation)")
plt.ylabel("Feature Name")
plt.title("Permutation Feature Importance for XGBoost")
plt.gca().invert_yaxis()  # Flip for better visualization
plt.show()


# %%
