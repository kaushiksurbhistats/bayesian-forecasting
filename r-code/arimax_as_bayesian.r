####################################################
# Fitting of ARIMAX Model under Bayesian approach
####################################################

# Install required packages
install.packages("bayesarimax")  # Bayesian ARIMAX model
install.packages("ggplot2")      # For plotting

# Load libraries
library(bayesarimax)
library(ggplot2)

# Step 1: Import your data
# Assume your data is in CSV format, and you have columns for Year, Yield, Max Temp, Min Temp, and Rainfall
data <- read.csv("path_to_your_data.csv")

# Convert the year to Date format and set it as the time series index
data_ts <- ts(data$Yield, start=c(1980), frequency=1)  # Adjust frequency for yearly data

# Step 2: Prepare your external regressors (weather data: Max Temp, Min Temp, Rainfall)
weather_data <- cbind(data$MaxTemp, data$MinTemp, data$Rainfall)
colnames(weather_data) <- c("MaxTemp", "MinTemp", "Rainfall")

# Step 3: Fit ARIMAX model using Bayesian estimation (MCMC)
# Specify the order of the ARIMAX model (p, d, q) and external regressors
model <- bayesarimax(y = data_ts, 
                     X = weather_data, 
                     order = c(1, 1, 1),  # Example ARIMA(1, 1, 1)
                     burnin = 1000,       # Number of burn-in iterations for MCMC
                     iter = 2000,         # Total number of iterations
                     thin = 1,            # Thinning parameter
                     verbose = TRUE)

# Step 4: Forecast for the next 5 years
forecast_horizon <- 5  # 5 years
forecast_results <- predict(model, n.ahead = forecast_horizon)

# Step 5: Plot comparison of observed vs predicted yield
# Assuming observed data is available for the forecast period
observed_data <- read.csv("path_to_observed_data.csv")
observed_yield <- observed_data$Yield  # Ensure this matches forecast period

# Combine predicted and observed values for comparison
comparison_df <- data.frame(
  Year = seq(from = max(data$Year), by = 1, length.out = forecast_horizon),
  Observed = observed_yield,
  Predicted = forecast_results$mean
)

# Plot observed vs predicted
ggplot(comparison_df, aes(x = Year)) +
  geom_line(aes(y = Observed, color = "Observed"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 1) +
  labs(title = "Observed vs Predicted Mustard Yield",
       x = "Year", y = "Mustard Yield") +
  scale_color_manual(values = c("Observed" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Step 6: Calculate the Galman Ratio (for forecasting accuracy)
galman_ratio <- mean(abs(forecast_results$mean - observed_yield) / observed_yield)
print(paste("Galman Ratio: ", round(galman_ratio, 4)))
