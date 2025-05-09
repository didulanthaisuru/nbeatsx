# -*- coding: utf-8 -*-
"""NBEATSx model for time series forecasting with additional features"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import HuberQLoss, MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Read your data
df = pd.read_excel('featured_shihara.xlsx', engine='openpyxl')
df["Date"] = pd.to_datetime(df["Date"])
nf_df = df

# Normalize the target variable
scaler = StandardScaler()
nf_df['Normalized_Balance'] = scaler.fit_transform(nf_df[['Normalized_Balance']])

# Feature engineering
nf_df['day_of_month'] = nf_df['Date'].dt.day
nf_df['month'] = nf_df['Date'].dt.month
nf_df['year'] = nf_df['Date'].dt.year
nf_df['is_weekend'] = nf_df['Date'].dt.dayofweek >= 5

# Calculate difference features
nf_df['balance_diff_1d'] = nf_df['Normalized_Balance'].diff(1)
nf_df['balance_diff_7d'] = nf_df['Normalized_Balance'].diff(7)
nf_df['balance_pct_change_1d'] = nf_df['Normalized_Balance'].pct_change(1)

# Calculate rolling features with min_periods to handle NaN values
nf_df['rolling_mean_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).mean()
nf_df['rolling_min_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).min()
nf_df['rolling_max_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).max()

# Exponential weighted features
nf_df['ewm_7d'] = nf_df['Normalized_Balance'].ewm(span=7, min_periods=1).mean()
nf_df['ewm_30d'] = nf_df['Normalized_Balance'].ewm(span=30, min_periods=1).mean()

# Additional feature engineering
nf_df['quarter'] = nf_df['Date'].dt.quarter
nf_df['day_of_week'] = nf_df['Date'].dt.dayofweek
nf_df['is_month_start'] = nf_df['Date'].dt.is_month_start.astype(int)
nf_df['is_month_end'] = nf_df['Date'].dt.is_month_end.astype(int)

# Advanced rolling features with min_periods
nf_df['rolling_std_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).std()
nf_df['rolling_skew_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).skew()
nf_df['rolling_kurt_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).kurt()

# Momentum indicators
nf_df['momentum_7d'] = nf_df['Normalized_Balance'] - nf_df['Normalized_Balance'].shift(7)
nf_df['momentum_14d'] = nf_df['Normalized_Balance'] - nf_df['Normalized_Balance'].shift(14)

# Volatility features
nf_df['volatility_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).std()
nf_df['volatility_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).std()

# Rate of change features
nf_df['roc_7d'] = nf_df['Normalized_Balance'].pct_change(7)
nf_df['roc_14d'] = nf_df['Normalized_Balance'].pct_change(14)

# Fill NaN values using the recommended methods
nf_df = nf_df.bfill().ffill()

# Update exogenous variables list
exogenous_vars = [col for col in nf_df.columns
                 if col not in ['Date', 'Normalized_Balance', 'unique_id', 'ds', 'y']]

# Data preparation: Split into train (80%), validation (10%), and test (10%)
train_size = int(len(nf_df) * 0.8)
val_size = int(len(nf_df) * 0.1)
train_df = nf_df.iloc[:train_size].copy()
val_df = nf_df.iloc[train_size:train_size+val_size].copy()
test_df = nf_df.iloc[train_size+val_size:].copy()

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Testing set size: {len(test_df)}")

# Handle missing values in all datasets
train_df = train_df.bfill().ffill()
val_df = val_df.bfill().ffill()
test_df = test_df.bfill().ffill()

# Set up the data in NeuralForecast format
train_data = train_df.copy()
train_data['unique_id'] = 'balance'
train_data = train_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

val_data = val_df.copy()
val_data['unique_id'] = 'balance'
val_data = val_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

test_data = test_df.copy()
test_data['unique_id'] = 'balance'
test_data = test_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

print("\nTraining data statistics:")
print(train_data['y'].describe())

horizon = 30
model = NBEATSx(
    h=horizon,                   
    input_size=180,             
    futr_exog_list=exogenous_vars,
    hist_exog_list=exogenous_vars,
    random_seed=42,
    scaler_type='standard',     
    learning_rate=0.001,        
    max_steps=1000,             
    batch_size=32,              
    loss=MAE(),                 # Changed to MAE loss
    valid_loss=MAE(),           # Added validation loss
    # Model architecture parameters
    stack_types=['identity', 'trend'],  
    n_blocks=[2, 2],            
    mlp_units=[[256, 256], [256, 256]],  
    dropout_prob_theta=0.1,     
    early_stop_patience_steps=10,
    val_check_steps=50          # Check validation every 50 steps
)

# Create the forecaster
forecaster = NeuralForecast(
    models=[model],
    freq='D'  
)

# Fit the model with validation
forecaster.fit(df=train_data, val_df=val_data)

# Generate forecasts
forecast_df = forecaster.predict(
    futr_df=test_data.iloc[:horizon]
)

print("\nForecast results:")
print(forecast_df.head())

# Extract actual values and forecasts
actual = test_data['y'].iloc[:horizon].values
forecast = forecast_df.loc[forecast_df['unique_id'] == 'balance', 'NBEATSx'].values

# Inverse transform the predictions and actual values
actual = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

# Calculate error metrics
mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100

# Additional metrics
r2_score = 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
explained_variance = np.var(forecast) / np.var(actual)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2_score:.4f}")
print(f"Explained Variance: {explained_variance:.4f}")

# Create a more detailed visualization
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Forecast
plt.subplot(2, 1, 1)
plt.plot(range(horizon), actual, label='Actual', marker='o', alpha=0.7)
plt.plot(range(horizon), forecast, label='NBEATSx Forecast', marker='x', alpha=0.7)
plt.fill_between(range(horizon), 
                 forecast - rmse, 
                 forecast + rmse, 
                 alpha=0.2, 
                 label='RMSE Range')
plt.title('30-Day Balance Forecast vs Actual')
plt.xlabel('Days')
plt.ylabel('Normalized Balance')
plt.legend()
plt.grid(True)

# Plot 2: Error Analysis
plt.subplot(2, 1, 2)
errors = actual - forecast
plt.plot(range(horizon), errors, label='Prediction Error', color='red', marker='o')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.fill_between(range(horizon), 
                 -rmse, 
                 rmse, 
                 alpha=0.2, 
                 label='RMSE Range')
plt.title('Prediction Error Analysis')
plt.xlabel('Days')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Analyze error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, alpha=0.7)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print error statistics
print("\nError Statistics:")
print(f"Mean Error: {np.mean(errors):.4f}")
print(f"Error Standard Deviation: {np.std(errors):.4f}")
print(f"Error Skewness: {pd.Series(errors).skew():.4f}")
print(f"Error Kurtosis: {pd.Series(errors).kurtosis():.4f}")