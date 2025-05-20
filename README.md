# Predict Calorie Expenditure - Kaggle Playground Series S5E5

This repository contains my submission for the Kaggle competition [Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5), where the objective is to predict calorie expenditure using biometric and activity-related features.

## 🧠 Objective

To build a regression model that accurately predicts calorie expenditure from features like age, sex, height, weight, heart rate, and body temperature.

## 📁 Contents

- `calories.ipynb` — Jupyter Notebook containing:
  - Exploratory data analysis (EDA)
  - Feature engineering
  - Model training and evaluation
  - Final predictions and submission generation

## 🔧 Tools and Libraries Used

- Python
- pandas & numpy — Data handling
- seaborn & matplotlib — Visualization
- scikit-learn — Modeling and metrics
- RandomForestRegressor — Final model used

## 📊 Approach

1. **Exploratory Data Analysis (EDA)**
   - Checked distributions, outliers, and feature correlations
   - Visualized relationships between predictors and the target variable (`Calories`)

2. **Feature Engineering**
   - Created a new feature: **BMI** = Weight / (Height in meters)^2
   - Encoded categorical variable `Sex` (male → 1, female → 0)

3. **Modeling**
   - Split training data into training and validation sets
   - Used **RandomForestRegressor** with 100 estimators
   - Evaluated performance using:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - Normalized MSE (NMSE)

4. **Final Model Performance (on validation set)**
   - MSE: 14.306842732279897
   - RMSE: 3.782438728159373
   - MAE: 2.2855284460450025
   - NMSE: 0.003672192514409179

5. **Submission**
   - Generated predictions for the test dataset
   - Saved predictions to a submission CSV file for upload to Kaggle
   - Last Submission Score: 0.06232

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook calories.ipynb
   ```
4. Download the dataset from Kaggle and place it in the proper location (`/kaggle/input/playground-series-s5e5/`).

## 📌 Notes

- This was an individual learning exercise on regression modeling and Kaggle workflows.
- Future improvements could include:
  - Hyperparameter tuning using GridSearchCV or Optuna
  - Trying out ensemble models (e.g., stacking)
  - Feature selection based on importance scores

## 📜 License

This project is for educational use only and complies with the rules of the Kaggle Playground Series.
