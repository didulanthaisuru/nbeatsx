# -*- coding: utf-8 -*-
"""NBEATSx model for balance forecasting with optimized parameters"""

import os
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Required packages:
# pip install neuralforecast
# pip install ipywidgets
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MAE

# Load and prepare data
ff = '/content/drive/MyDrive/Colab Notebooks/featured_shihara.xlsx'
df = pd.read_excel(ff, engine='openpyxl')
df["Date"] = pd.to_datetime(df["Date"])

# Data preprocessing
nf_df = df.copy()

# Additional feature engineering
# Time-based features
nf_df['day_of_month'] = nf_df['Date'].dt.day
nf_df['month'] = nf_df['Date'].dt.month
nf_df['year'] = nf_df['Date'].dt.year
nf_df['quarter'] = nf_df['Date'].dt.quarter
nf_df['is_weekend'] = nf_df['Date'].dt.dayofweek >= 5
nf_df['is_month_start'] = nf_df['Date'].dt.is_month_start.astype(int)
nf_df['is_month_end'] = nf_df['Date'].dt.is_month_end.astype(int)

# Advanced rolling features with min_periods to handle NaN values
for window in [7, 14, 30]:
    nf_df[f'rolling_mean_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).mean()
    nf_df[f'rolling_std_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).std()
    nf_df[f'rolling_min_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).min()
    nf_df[f'rolling_max_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).max()
    nf_df[f'rolling_skew_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).skew()
    nf_df[f'rolling_kurt_{window}d'] = nf_df['Normalized_Balance'].rolling(window=window, min_periods=1).kurt()

# Momentum and rate of change features
nf_df['balance_diff_1d'] = nf_df['Normalized_Balance'].diff(1)
nf_df['balance_diff_7d'] = nf_df['Normalized_Balance'].diff(7)
nf_df['balance_pct_change_1d'] = nf_df['Normalized_Balance'].pct_change(1).fillna(0).clip(-1, 1)
nf_df['balance_pct_change_7d'] = nf_df['Normalized_Balance'].pct_change(7).fillna(0).clip(-1, 1)

# Exponential weighted features
nf_df['ewm_7d'] = nf_df['Normalized_Balance'].ewm(span=7, min_periods=1).mean()
nf_df['ewm_14d'] = nf_df['Normalized_Balance'].ewm(span=14, min_periods=1).mean()
nf_df['ewm_30d'] = nf_df['Normalized_Balance'].ewm(span=30, min_periods=1).mean()

# Fill NaN values
nf_df = nf_df.fillna(method='bfill').fillna(method='ffill')

# Normalize all features
scaler = StandardScaler()
feature_cols = [col for col in nf_df.columns if col not in ['Date', 'Normalized_Balance']]
nf_df[feature_cols] = scaler.fit_transform(nf_df[feature_cols])

# Data split (90% train, 10% test) - Using more training data for better accuracy
train_size = int(len(nf_df) * 0.9)
train_df = nf_df.iloc[:train_size].copy()
test_df = nf_df.iloc[train_size:].copy()

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# Prepare data for NeuralForecast
train_data = train_df.copy()
train_data['unique_id'] = 'balance'
train_data = train_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

test_data = test_df.copy()
test_data['unique_id'] = 'balance'
test_data = test_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

# Define exogenous variables
exogenous_vars = [col for col in train_data.columns
                 if col not in ['ds', 'y', 'unique_id']]

print(f"\nUsing {len(exogenous_vars)} exogenous variables")

# Define forecast horizon
horizon = 30

# Create and train the NBEATSx model with optimized parameters for maximum accuracy
model = NBEATSx(
    h=horizon,                   
    input_size=160,              # Increased for better pattern recognition
    futr_exog_list=exogenous_vars,
    hist_exog_list=exogenous_vars,
    random_seed=42,
    scaler_type='standard',     
    learning_rate=0.001,        # Optimal learning rate
    max_steps=1000,             # Increased for better convergence
    batch_size=1,               # Using batch size of 1 for maximum accuracy
    stack_types=['identity', 'trend', 'seasonality'],  # Using all stack types
    n_blocks=[3, 3, 3],         # Increased number of blocks
    mlp_units=[[256, 256, 256], [256, 256, 256], [256, 256, 256]],  # Increased network capacity
    dropout_prob_theta=0.1,     # Moderate dropout for regularization
    early_stop_patience_steps=50,  # Increased patience for better convergence
    val_check_steps=20,
    loss=MAE(),                 # Using MAE loss for better robustness
    valid_loss=MAE(),           # Using MAE for validation
    optimizer=AdamW,            # Using AdamW optimizer
    optimizer_kwargs={
        'weight_decay': 1e-5,   # Reduced weight decay for better learning
        'betas': (0.9, 0.999)
    },
    gradient_clip_val=0.5,      # Reduced gradient clipping for better learning
    lr_scheduler=OneCycleLR,    # Using OneCycleLR scheduler class
    lr_scheduler_kwargs={
        'max_lr': 0.001,
        'pct_start': 0.3,
        'div_factor': 25.0,
        'final_div_factor': 1000.0,
        'total_steps': 1000,    # Set to max_steps
        'anneal_strategy': 'cos'  # Using cosine annealing
    }
)

# Create the forecaster with verbose output
forecaster = NeuralForecast(
    models=[model],
    freq='D',
    verbose=True
)

# Print data shapes and statistics before training
print("\nTraining Data Statistics:")
print(f"Training data shape: {train_data.shape}")
print(f"Number of features: {len(exogenous_vars)}")
print("\nFeature Statistics:")
print(train_data[exogenous_vars].describe())

# Fit the model with appropriate validation size
forecaster.fit(
    df=train_data,
    val_size=30  # Set to horizon size
)

# Generate forecasts
forecast_df = forecaster.predict(
    futr_df=test_data.iloc[:horizon]
)

# Extract actual values and forecasts
actual = test_data['y'].iloc[:horizon].values
forecast = forecast_df.loc[forecast_df['unique_id'] == 'balance', 'NBEATSx'].values

# Print raw values for debugging
print("\nFirst few actual values:", actual[:5])
print("First few forecast values:", forecast[:5])

# Calculate error metrics
mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100
r2 = r2_score(actual, forecast)
explained_variance = np.var(forecast) / np.var(actual)

# Create comprehensive visualization
plt.style.use('default')  # Using default matplotlib style
fig = plt.figure(figsize=(15, 12))

# Plot 1: Actual vs Forecast
plt.subplot(2, 2, 1)
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
plt.subplot(2, 2, 2)
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

# Plot 3: Error Distribution
plt.subplot(2, 2, 3)
plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)

# Plot 4: Scatter Plot
plt.subplot(2, 2, 4)
plt.scatter(actual, forecast, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Print comprehensive metrics
print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"Explained Variance: {explained_variance:.4f}")

print("\nError Statistics:")
print(f"Mean Error: {np.mean(errors):.4f}")
print(f"Error Standard Deviation: {np.std(errors):.4f}")
print(f"Error Skewness: {pd.Series(errors).skew():.4f}")
print(f"Error Kurtosis: {pd.Series(errors).kurtosis():.4f}")

plt.show()

# Save the model
forecaster.save('nbeatsx_optimized_model')
print("\nModel saved as 'nbeatsx_optimized_model'") 