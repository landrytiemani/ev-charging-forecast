# EV Charging Demand Forecasting in Boulder, CO

This project uses an ARIMA time series model to forecast monthly electric vehicle (EV) charging demand in Boulder, Colorado, based on historical station usage data.

## Dataset

Use the cleaned charging dataset:
```
data/Electric_Vehicle_Charging_Station_Data.csv from https://open-data.bouldercolorado.gov/datasets/95992b3938be4622b07f0b05eba95d4c_0/explore
```

(Uploaded from Boulder’s open data portal)

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python main.py
```

3. Output:
- Forecast plot at `visuals/forecast_plot.png`
- Trained model at `models/arima_model.pkl`

## 📏 Evaluation

- Test RMSE (Root Mean Squared Error) is computed using the last 12 months.
- This helps assess forecast accuracy before projecting future demand.

## Notes

- Data is aggregated monthly by total kWh charged.
- This helps identify trends for energy planning and infrastructure scaling.

