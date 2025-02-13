##########################################################
#   Fitting using ARIMA Model under classical approach.
##########################################################

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import ljungbox
from scipy.stats import shapiro

# Step 1: Import Data
# Replace 'your_file.csv' with your actual file path or dataset
data = pd.read_csv('your_file.csv', parse_dates=['Date'], index_col='Date')
data = data.asfreq('Y')  # Ensure data is yearly (change as per your data frequency)
y = data['Yield']  # Replace 'Yield' with the column name of your time series

# Plot original data
plt.figure(figsize=(10, 6))
plt.plot(y, label='Original Data')
plt.title('Mustard Yield Data')
plt.legend()
plt.show()

# Step 2: Stationarity Check (ADF Test)
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")

adf_test(y)

# Step 3: Differencing to Make Data Stationary
y_diff = y.diff().dropna()
plt.figure(figsize=(10, 6))
plt.plot(y_diff, label='Differenced Data')
plt.title('Differenced Data')
plt.legend()
plt.show()

adf_test(y_diff)

# Step 4: ACF and PACF Plots
plot_acf(y_diff, lags=20)
plt.title('ACF Plot')
plt.show()

plot_pacf(y_diff, lags=20)
plt.title('PACF Plot')
plt.show()

# Step 5: ARIMA Model Fitting
# Select (p, d, q) based on ACF, PACF, and differencing results
p, d, q = 1, 1, 1  # Replace with optimal values
model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Step 6: Model Evaluation (AIC and BIC)
print("AIC:", model_fit.aic)
print("BIC:", model_fit.bic)

# Step 7: Residual Diagnostics
residuals = model_fit.resid

# Residual Plot
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()

# Shapiro-Wilk Test for Normality
stat, p = shapiro(residuals)
print('Shapiro-Wilk Test Statistic:', stat)
print('p-value:', p)
if p > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")

# Ljung-Box Test for Autocorrelation
ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(ljung_box_result)

# Step 8: Forecasting for 5 Years
forecast = model_fit.get_forecast(steps=5)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(y, label='Historical Data')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('5-Year Mustard Yield Forecast')
plt.legend()
plt.show()

# Step 9: Comparison Plot (Original vs Predicted Throughout Study Period)
y_pred = model_fit.fittedvalues
plt.figure(figsize=(12, 6))
plt.plot(y, label='Original Data', color='blue')
plt.plot(y_pred, label='Fitted Data', color='orange')
plt.title('Original vs Predicted Yield Throughout Study Period')
plt.legend()
plt.show()
