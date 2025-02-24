import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Generate synthetic data for building age and claim frequency
np.random.seed(42)
n = 100
building_age = np.random.uniform(5, 50, n)  # Age between 5 and 50 years
claim_frequency = 1.5 - 0.05 * building_age + 0.001 * building_age**2 + np.random.normal(0, 0.2, n)  # Non-linear relationship

# Create a DataFrame
df = pd.DataFrame({'Building_Age': building_age, 'Claim_Frequency': claim_frequency})

# Fit a GLM with a simple linear predictor
model_linear = smf.glm("Claim_Frequency ~ Building_Age", data=df, family=sm.families.Gaussian()).fit()

# Get model predictions
df['Predicted_Linear'] = model_linear.predict(df)

# Compute partial residuals using equation:
df['Partial_Residual'] = df['Claim_Frequency'] - df['Predicted_Linear'] + model_linear.params['Building_Age'] * df['Building_Age']

# Plot partial residuals
plt.figure(figsize=(8, 6))
plt.scatter(df['Building_Age'], df['Partial_Residual'], label="Partial Residuals", color="blue", alpha=0.6)
plt.plot(df['Building_Age'], model_linear.params['Building_Age'] * df['Building_Age'], color='red', label="Linear Fit", linewidth=2)
plt.xlabel("Building Age")
plt.ylabel("Partial Residuals")
plt.title("Partial Residual Plot: Checking Linearity")
plt.legend()
plt.show()
