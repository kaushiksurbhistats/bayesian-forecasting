# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

# Step 1: Load Data
# Replace 'your_data.csv' with your actual file path
data = pd.read_csv('your_data.csv', parse_dates=['Date'], index_col='Date')

# Dependent variable: Yield
y = data['Yield']  # Replace 'Yield' with the actual column name

# Exogenous variables: Selected features
# Replace 'selected_features.csv' with the file containing selected variable names
selected_features = pd.read_csv('selected_features.csv')['Selected Features'].tolist()
X = data[selected_features]

# Step 2: Train-Test Split
# Use the entire data for fitting ARIMAX and reserve the last 5 years for forecasting
train_end = len(data) - 5  # Adjust as needed
y_train = y.iloc[:train_end]
X_train = X.iloc[:train_end]

# Step 3: ARIMAX Model Fitting
# Set ARIMA order (p, d, q) based on your data analysis
p, d, q = 1, 1, 1  # Adjust based on prior ACF/PACF analysis
model = SARIMAX(y_train, exog=X_train, order=(p, d, q))
model_fit = model.fit(disp=False)

# Print model summary
print(model_fit.summary())

# Step 4: Residual Diagnostics
residuals = model_fit.resid

# Shapiro-Wilk Test for Normality
stat, p_value = shapiro(residuals)
print('Shapiro-Wilk Test Statistic:', stat)
print('p-value:', p_value)
if p_value > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")

# Ljung-Box Test for Autocorrelation
ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(ljung_box_result)

# Step 5: Forecasting for 5 Years
X_forecast = X.iloc[train_end:]  # Exogenous variables for the forecast period
forecast = model_fit.get_forecast(steps=5, exog=X_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(y, label='Historical Data')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('5-Year Forecast with ARIMAX')
plt.legend()
plt.show()

# Step 6: Comparison Plot (Original vs Predicted Throughout Study Period)
# In-sample predictions
y_pred_in_sample = model_fit.fittedvalues

# Plot original vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y, label='Original Data', color='blue')
plt.plot(y_pred_in_sample, label='Fitted Data', color='orange')
plt.title('Original vs Predicted (In-Sample)')
plt.legend()
plt.show()
