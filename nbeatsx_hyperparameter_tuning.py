import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import itertools
import json
from datetime import datetime

# Read and preprocess data (reusing the same preprocessing steps)
df = pd.read_excel('featured_shihara.xlsx', engine='openpyxl')
df["Date"] = pd.to_datetime(df["Date"])
nf_df = df

# Normalize the target variable
scaler = StandardScaler()
nf_df['Normalized_Balance'] = scaler.fit_transform(nf_df[['Normalized_Balance']])

# Feature engineering (reusing the same features)
nf_df['day_of_month'] = nf_df['Date'].dt.day
nf_df['month'] = nf_df['Date'].dt.month
nf_df['year'] = nf_df['Date'].dt.year
nf_df['is_weekend'] = nf_df['Date'].dt.dayofweek >= 5
nf_df['balance_diff_1d'] = nf_df['Normalized_Balance'].diff(1)
nf_df['balance_diff_7d'] = nf_df['Normalized_Balance'].diff(7)
nf_df['balance_pct_change_1d'] = nf_df['Normalized_Balance'].pct_change(1)
nf_df['rolling_mean_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).mean()
nf_df['rolling_min_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).min()
nf_df['rolling_max_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).max()
nf_df['ewm_7d'] = nf_df['Normalized_Balance'].ewm(span=7, min_periods=1).mean()
nf_df['ewm_30d'] = nf_df['Normalized_Balance'].ewm(span=30, min_periods=1).mean()
nf_df['quarter'] = nf_df['Date'].dt.quarter
nf_df['day_of_week'] = nf_df['Date'].dt.dayofweek
nf_df['is_month_start'] = nf_df['Date'].dt.is_month_start.astype(int)
nf_df['is_month_end'] = nf_df['Date'].dt.is_month_end.astype(int)
nf_df['rolling_std_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).std()
nf_df['rolling_skew_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).skew()
nf_df['rolling_kurt_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).kurt()
nf_df['momentum_7d'] = nf_df['Normalized_Balance'] - nf_df['Normalized_Balance'].shift(7)
nf_df['momentum_14d'] = nf_df['Normalized_Balance'] - nf_df['Normalized_Balance'].shift(14)
nf_df['volatility_7d'] = nf_df['Normalized_Balance'].rolling(window=7, min_periods=1).std()
nf_df['volatility_14d'] = nf_df['Normalized_Balance'].rolling(window=14, min_periods=1).std()
nf_df['roc_7d'] = nf_df['Normalized_Balance'].pct_change(7)
nf_df['roc_14d'] = nf_df['Normalized_Balance'].pct_change(14)

# Fill NaN values
nf_df = nf_df.bfill().ffill()

# Get exogenous variables
exogenous_vars = [col for col in nf_df.columns
                 if col not in ['Date', 'Normalized_Balance', 'unique_id', 'ds', 'y']]

# Data preparation
train_size = int(len(nf_df) * 0.8)
val_size = int(len(nf_df) * 0.1)
train_df = nf_df.iloc[:train_size].copy()
val_df = nf_df.iloc[train_size:train_size+val_size].copy()
test_df = nf_df.iloc[train_size+val_size:].copy()

# Prepare data in NeuralForecast format
train_data = train_df.copy()
train_data['unique_id'] = 'balance'
train_data = train_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

val_data = val_df.copy()
val_data['unique_id'] = 'balance'
val_data = val_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

test_data = test_df.copy()
test_data['unique_id'] = 'balance'
test_data = test_data.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})

# Define hyperparameter grid
param_grid = {
    'input_size': [90, 120, 180],
    'learning_rate': [0.0001, 0.001, 0.01],
    'max_steps': [500, 1000, 1500],
    'batch_size': [16, 32, 64],
    'n_blocks': [[2, 2], [3, 3], [4, 4]],
    'mlp_units': [
        [[128, 128], [128, 128]],
        [[256, 256], [256, 256]],
        [[512, 512], [512, 512]]
    ],
    'dropout_prob_theta': [0.1, 0.2, 0.3]
}

# Function to evaluate model with given parameters
def evaluate_model(params):
    horizon = 30
    model = NBEATSx(
        h=horizon,
        input_size=params['input_size'],
        futr_exog_list=exogenous_vars,
        hist_exog_list=exogenous_vars,
        random_seed=42,
        scaler_type='standard',
        learning_rate=params['learning_rate'],
        max_steps=params['max_steps'],
        batch_size=params['batch_size'],
        loss=MAE(),
        valid_loss=MAE(),
        stack_types=['identity', 'trend'],
        n_blocks=params['n_blocks'],
        mlp_units=params['mlp_units'],
        dropout_prob_theta=params['dropout_prob_theta'],
        early_stop_patience_steps=10,
        val_check_steps=50
    )

    forecaster = NeuralForecast(
        models=[model],
        freq='D'
    )

    # Fit the model
    forecaster.fit(df=train_data, val_df=val_data)

    # Generate forecasts
    forecast_df = forecaster.predict(
        futr_df=test_data.iloc[:horizon]
    )

    # Extract actual values and forecasts
    actual = test_data['y'].iloc[:horizon].values
    forecast = forecast_df.loc[forecast_df['unique_id'] == 'balance', 'NBEATSx'].values

    # Inverse transform
    actual = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

    # Calculate metrics
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100

    return {
        'params': params,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    }

# Generate all parameter combinations
param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# Store results
results = []
best_mae = float('inf')
best_params = None

# Run grid search
print("Starting hyperparameter tuning...")
for i, params in enumerate(param_combinations):
    print(f"\nTesting combination {i+1}/{len(param_combinations)}")
    print(f"Parameters: {params}")
    
    try:
        result = evaluate_model(params)
        results.append(result)
        
        # Update best parameters if better MAE is found
        if result['metrics']['mae'] < best_mae:
            best_mae = result['metrics']['mae']
            best_params = params
            
        print(f"Metrics: MAE={result['metrics']['mae']:.4f}, RMSE={result['metrics']['rmse']:.4f}, MAPE={result['metrics']['mape']:.2f}%")
    except Exception as e:
        print(f"Error with parameters {params}: {str(e)}")
        continue

# Save results to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'nbeatsx_tuning_results_{timestamp}.json'

with open(results_file, 'w') as f:
    json.dump({
        'all_results': results,
        'best_params': best_params,
        'best_mae': best_mae
    }, f, indent=4)

print("\nHyperparameter tuning completed!")
print(f"\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"\nBest MAE: {best_mae:.4f}")
print(f"\nFull results saved to {results_file}") 