# Loss Ratio Prediction Using Tweedie Regression & XGBoost

This repository contains an end-to-end implementation of **Tweedie Regression (GLM) and XGBoost Tweedie models** for **insurance loss ratio prediction**. The project focuses on **model selection, feature engineering, hyperparameter tuning, and evaluation** using real-world structured data.

## Project Overview

The goal is to build a predictive model for **insurance loss ratios** using:
- **Generalized Linear Models (GLMs) with Tweedie distribution**
- **XGBoost with Tweedie loss function**
- **Feature selection** using **Boruta** and **Lasso regularization**
- **Hyperparameter tuning** for both Tweedie GLM and XGBoost
- **Evaluation using RMSE** on validation data
- **Final predictions on test data**, with appropriate post-processing

## Data Description

The dataset consists of structured insurance-related features, including:

- **Policyholder attributes**: Age, marital status, employment details
- **Claim information**: Incurred claim costs, accident details
- **Financial indicators**: Earned premiums, weekly wages
- **Derived variables**: Loss ratio, interaction terms for feature enhancement

The dataset is split into:
- **Training (80%)**
- **Validation (20%)**
- **A separate test set** for final evaluation.

## Modeling Approach

### Tweedie Regression (GLM)
- Implemented **Tweedie GLM** with multiple **variance power values** (1.1 to 1.9)
- Used **log-link function** for better distribution fit
- Evaluated model performance using **RMSE**

### XGBoost Tweedie Regression
- Trained **XGBoost with Tweedie loss function** to capture **nonlinear relationships**
- Applied **early stopping** (best iteration: 81 rounds)
- Tuned hyperparameters, including:
  - `max_depth`, `eta`, `subsample`, `colsample_bytree`, `min_child_weight`
- **Final RMSE: 0.4071** (best model)

## Model Performance

| Model                  | Validation RMSE |
|------------------------|----------------|
| Tweedie GLM (best)    | 0.4667         |
| Initial XGBoost       | 0.4170         |
| Tuned XGBoost (final) | **0.4071**     |

XGBoost outperforms Tweedie GLM by **capturing nonlinear interactions and complex dependencies**.

## Feature Engineering & Selection

- **Boruta** was used to identify the most influential variables
- **Lasso regression** was applied to refine feature selection
- **Interaction term** `(EarnedPremiums × HoursWorkedPerWeek)` was introduced for enhanced model expressiveness

## Visualization

- **Comparison of train, validation, and test distributions** to ensure consistency
- **Residual analysis** to verify model assumptions
- **Feature importance plots** to interpret key predictors

## Installation & Setup

### Requirements
Ensure you have **R (>= 4.0)** and install the required libraries:

```r
install.packages(c("tidyverse", "lubridate", "caret", "xgboost", "glmnet", "Boruta"))


### Clone the Repository:

git clone https://github.com/your-repo/loss-ratio-prediction.git
cd loss-ratio-prediction

