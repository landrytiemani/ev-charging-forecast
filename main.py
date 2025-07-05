from src.preprocess import load_ev_data
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# === PATHS ===
data_path = "data/Electric_Vehicle_Charging_Station_Data.csv"
model_path = "models/auto_arima_model.pkl"
plot_path = "visuals/forecast_plot.png"
forecast_csv_path = "output/forecast_24months.csv"

# === CREATE FOLDERS ===
os.makedirs("models", exist_ok=True)
os.makedirs("visuals", exist_ok=True)
os.makedirs("output", exist_ok=True)

# === STEP 1: LOAD CLEANED DATA ===
df = load_ev_data(data_path)
series = df['energy_kwh']

# === STEP 2: TRAIN / TEST SPLIT ===
train = series[:-12]
test = series[-12:]

# === STEP 3: AUTO ARIMA MODELING ===
print("🔍 Running auto_arima...")
model = auto_arima(train, seasonal=True, m=12,
                   trace=True, error_action='ignore',
                   suppress_warnings=True, stepwise=True)

# === STEP 4: EVALUATE ON TEST SET ===
model.fit(train)
forecast_test = model.predict(n_periods=12)
rmse = np.sqrt(mean_squared_error(test, forecast_test))
print(f"✅ Test RMSE: {rmse:.2f} kWh")

# === STEP 5: REFIT ON FULL DATA & FORECAST 24 MONTHS ===
model.fit(series)
forecast_24, confint = model.predict(n_periods=24, return_conf_int=True)
forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1),
                               periods=24, freq='MS')

# === STEP 6: SAVE FORECAST TO CSV ===
forecast_df = pd.DataFrame({
    'date': forecast_index,
    'forecast_kwh': forecast_24,
    'lower_ci': confint[:, 0],
    'upper_ci': confint[:, 1]
})
forecast_df.to_csv(forecast_csv_path, index=False)
print(f"📁 Forecast saved to: {forecast_csv_path}")

# === STEP 7: PLOT FORECAST ===
plt.figure(figsize=(14, 6))
plt.plot(series.index, series.values, label='Observed', color='blue')

# Forecast line
plt.plot(forecast_index, forecast_24, label='Forecast (auto_arima)', color='black', linestyle='--', linewidth=2)

# Confidence intervals
plt.fill_between(forecast_index, confint[:, 0], confint[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')

# Labels and layout
plt.title(f"EV Charging Energy Forecast (auto_arima RMSE={rmse:.2f} kWh)")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.legend()

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

# Save and show
plt.savefig(plot_path)
plt.show()

