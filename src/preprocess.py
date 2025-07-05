import pandas as pd
def load_ev_data(path):
    df = pd.read_csv(path)

    # Force conversion to datetime
    df['Start_Date___Time'] = pd.to_datetime(df['Start_Date___Time'], errors='coerce')

    # Drop bad rows
    df.dropna(subset=['Start_Date___Time', 'Energy__kWh_'], inplace=True)

    # DEBUG
    print("🧪 Sample:", df['Start_Date___Time'].iloc[0])
    print("🧪 Type:", type(df['Start_Date___Time'].iloc[0]))

    # Monthly aggregation
    df['Start_Date'] = df['Start_Date___Time'].dt.to_period('M').dt.to_timestamp()
    df_monthly = df.groupby('Start_Date')['Energy__kWh_'].sum().reset_index()
    df_monthly.rename(columns={'Start_Date': 'date', 'Energy__kWh_': 'energy_kwh'}, inplace=True)
    df_monthly.set_index('date', inplace=True)
    df_monthly = df_monthly.asfreq('MS').fillna(0)

    return df_monthly

