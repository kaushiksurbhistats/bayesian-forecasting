import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset (replace 'data.csv' with your actual file)
# Assuming your dataset has a 'Year' column and a 'Yield' column
data = pd.read_csv('data.csv')
years = data['Year']
yields = data['Yield']

# Bayesian ARIMA Model
def bayesian_arima(y, order=(1, 0, 1)):
    p, d, q = order

    with pm.Model() as model:
        # Priors
        phi = pm.Normal('phi', mu=0, sigma=1)  # AR term
        theta = pm.Normal('theta', mu=0, sigma=1)  # MA term
        sigma = pm.InverseGamma('sigma', alpha=2, beta=1)  # Error term
        
        # ARIMA process
        y_hat = pm.ARIMA('y_hat', phi=phi, theta=theta, sigma=sigma, observed=y)
        
        # MCMC sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)
    
    return trace, model

# Fit the Bayesian ARIMA model
order = (1, 0, 1)  # ARIMA(1, 0, 1)
trace, model = bayesian_arima(yields, order=order)

# Convergence Check
print(az.rhat(trace))  # Gelman-Rubin statistic

# Forecasting
def forecast(trace, steps=5):
    phi_samples = trace.posterior['phi'].values.flatten()
    theta_samples = trace.posterior['theta'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()
    
    y_forecast = []
    y_last = yields[-1]
    
    for i in range(steps):
        ar_term = phi_samples * y_last
        ma_term = np.random.normal(loc=0, scale=sigma_samples)
        y_new = ar_term + ma_term
        y_forecast.append(np.mean(y_new))
        y_last = y_new.mean()
    
    return y_forecast

forecasted_values = forecast(trace, steps=5)

# Plot observed vs. predictive yield
forecast_years = np.arange(years.max() + 1, years.max() + 6)
plt.figure(figsize=(10, 6))
plt.plot(years, yields, label='Observed Yield', marker='o')
plt.plot(forecast_years, forecasted_values, label='Forecasted Yield', marker='x')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Observed vs Forecasted Yield')
plt.legend()
plt.show()
