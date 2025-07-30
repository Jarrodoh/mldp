# Earthquake Magnitude Prediction Project

## Overview
This project predicts the magnitude of earthquakes using machine learning, based on global seismic data from 2000–2025. It includes a Jupyter notebook for data analysis and modeling, and a Streamlit app for interactive predictions and data exploration.

## Files in This Repository

### 1. `machine_learning_notebook.ipynb`
- **Purpose:** Main notebook for data analysis, feature engineering, regression modeling, and interpretation.
- **Contents & Steps:**
  1. **Data Loading & EDA:** Loads the earthquake dataset and explores distributions, correlations, and geographic patterns.
  2. **Data Cleaning & Preprocessing:** Handles missing values, removes duplicates, detects and treats outliers, corrects data types, and validates geographic data.
  3. **Feature Engineering:** Creates new features (e.g., interaction terms, time-based features, magnitude categories) to improve model performance and interpretability.
  4. **Regression Modeling:** Trains and tunes multiple regression models (RandomForest, HistGradientBoosting, Ridge, Lasso, LinearSVR), compares their performance, and blends models for robustness.
  5. **Model Evaluation:** Reports accuracy metrics (RMSE, MAE, R², etc.), analyzes residuals, and visualizes results with plots (predicted vs actual, residuals, calibration, learning curve).
  6. **Feature Importance & Interpretation:** Explains which features are most impactful, why RandomForest is used for feature importance, and interprets the top features with friendly, example-based explanations.
  7. **Deployment:** Saves the best model for use in the Streamlit app.

### 2. `app.py`
- **Purpose:** Streamlit web app for interactive earthquake magnitude prediction and data visualization.
- **Contents & Features:**
  - Loads the trained model and expects user input for all relevant features.
  - Provides sliders and dropdowns for easy input, with tooltips and explanations for each feature.
  - Predicts earthquake magnitude instantly and displays results in a modern, visually appealing UI.
  - Includes sidebar navigation, project description, and educational content.
  - Visualizes earthquake locations on a map, with filtering by magnitude and depth.
  - Contains a friendly, example-based guide explaining each feature and why it matters.

### 3. `usgs_earthquake_data_2000_2025.csv`
- **Purpose:** The raw dataset of global seismic events used for analysis and modeling.
- **Source:** [Kaggle: Global Seismic Events 2000–2025](https://www.kaggle.com/datasets/pulastya/global-seismic-events-20002025)

## Project Workflow (Notebook Steps)
1. **Load Data:** Import the earthquake dataset and inspect its structure.
2. **Exploratory Data Analysis (EDA):** Visualize distributions, correlations, and geographic patterns to understand the data and identify issues.
3. **Data Cleaning:** Remove or impute missing values, eliminate duplicates, treat outliers, and ensure correct data types.
4. **Feature Engineering:** Create new features (e.g., interaction terms, time-based features, magnitude categories) to capture important patterns and improve model accuracy.
5. **Model Training & Tuning:** Train multiple regression models, tune hyperparameters, and compare their performance using robust metrics.
6. **Model Evaluation:** Analyze residuals, calibration, and learning curves to assess model quality and generalization.
7. **Feature Importance & Interpretation:** Identify and explain the most impactful features, using both original and engineered features, with clear, user-friendly explanations.
8. **Deployment:** Save the best-performing model for use in the Streamlit app.

## Dataset
- **Name:** Global Seismic Events 2000–2025
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/pulastya/global-seismic-events-20002025)
- **Description:** Contains global earthquake records with event details, locations, magnitudes, and measurement quality indicators.

## How to Use
1. **Run the Notebook:** Follow the steps in `machine_learning_notebook.ipynb` to analyze the data, engineer features, train models, and interpret results.
2. **Launch the App:** Run `app.py` with Streamlit to interactively predict earthquake magnitudes and explore the data.
3. **Explore & Learn:** Use the app’s feature guide and visualizations to understand what each input means and how it affects predictions.

---

For questions or improvements, feel free to reach out or contribute!
