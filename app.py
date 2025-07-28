import streamlit as st
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, time

st.set_page_config(page_title="Power Demand Predictor", layout="wide")

st.title("⚡ Power Demand Predictor")
st.markdown("Predicts next day's electricity consumption for a single district using weather forecast data.")
st.markdown('---')

# Load cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_power_data.csv")

df = load_data()

# Split features and target
X = df.drop(columns=["POWER_DEMAND"])
y = df["POWER_DEMAND"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log transform for training
y_train_log = np.log1p(y_train)

# Train model
model = XGBRegressor(n_estimators=8)
model.fit(X_train, y_train_log)

st.sidebar.header("🌍 Location Input")
import requests
import ipywidgets as widgets
from IPython.display import display

# List of major cities in Pakistan
import requests
import ipywidgets as widgets
from IPython.display import display, clear_output

# Data for major cities in Pakistan
per_capita_demand_kwh = {
    # Dehli: {population: 22.3M, per_capita_kwh: 1075}, Pakistan 606
    "Karachi": {"population": 18868021, "per_capita_kwh": 1300},
    "Lahore": {"population": 13004135, "per_capita_kwh": 1200},
    "Islamabad": {"population": 1108872, "per_capita_kwh": 1100},
    "Peshawar": {"population": 1905975, "per_capita_kwh": 850},
    "Quetta": {"population": 1565546, "per_capita_kwh": 700},
}
# Dropdown for city selection
selected_city = st.sidebar.selectbox("Select a City", per_capita_demand_kwh.keys())
capita_demand = per_capita_demand_kwh[selected_city]["per_capita_kwh"]
population = per_capita_demand_kwh[selected_city]["population"]

# Function to get lat/lon from Nominatim
def get_coordinates(city_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': city_name + ", Pakistan",
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'GeoCoder-Pakistan-Cities/1.0 (syedsharjeel321@gmail.com)'
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if data:
        lat = float(data[0]['lat'])
        lon = float(data[0]['lon'])
        return lat, lon
    else:
        return None, None

# Button to fetch coordinates
if st.sidebar.button("Predict Power Demand"):
    lat, lon = get_coordinates(selected_city)
    if lat and lon:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m", "apparent_temperature", "dew_point_2m", "relative_humidity_2m",
                "precipitation_probability", "wind_gusts_10m", "wind_speed_10m",
                "pressure_msl", "cloud_cover", "visibility", "uv_index", "rain"
            ],
            "timezone": "auto"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        # Create weather DataFrame
        weather_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "feelslike": hourly.Variables(1).ValuesAsNumpy(),
            "dew": hourly.Variables(2).ValuesAsNumpy(),
            "humidity": hourly.Variables(3).ValuesAsNumpy(),
            "precipprob": hourly.Variables(4).ValuesAsNumpy(),
            "windgust": hourly.Variables(5).ValuesAsNumpy(),
            "windspeed": hourly.Variables(6).ValuesAsNumpy(),
            "sealevelpressure": hourly.Variables(7).ValuesAsNumpy(),
            "cloudcover": hourly.Variables(8).ValuesAsNumpy(),
            "visibility": hourly.Variables(9).ValuesAsNumpy(),
            "uvindex": hourly.Variables(10).ValuesAsNumpy(),
            "preciptype_rain": hourly.Variables(11).ValuesAsNumpy()
        }
        weather_df = pd.DataFrame(weather_data)
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

        # Filter next day's data
        now = pd.Timestamp.now(tz="UTC")
        next_day = now.normalize() + pd.Timedelta(days=1)
        day_after = next_day + pd.Timedelta(days=1)
        next_day_weather = weather_df[(weather_df["datetime"] >= next_day) & (weather_df["datetime"] < day_after)].copy()

        if next_day_weather.empty:
            st.error("⚠️ Weather data for the next day is not available. Try again later.")
        else:
            # Prepare data for prediction
            X_input = next_day_weather.drop(columns=["datetime"])
            y_log_pred = model.predict(X_input)
            y_pred = np.expm1(y_log_pred)

            adjustment_factor = ((population * 0.606) / (22300000 * 1.075))
            next_day_weather["Predicted_Power_Demand"] = y_pred*adjustment_factor
            total_demand = y_pred.sum()
            
            peak_row = next_day_weather.iloc[np.argmax(y_pred)]
            peak_demand = peak_row['Predicted_Power_Demand']
            
            adjusted_total_demand = total_demand * adjustment_factor
            adjusted_peak_demand = peak_demand
            
            st.markdown(f"🔋 **Total Predicted Demand (Next Day):** {adjusted_total_demand:.2f} MWh")
            st.markdown(f"⏰ **Peak Hour:** {peak_row['datetime']} with Demand: {adjusted_peak_demand:.2f} MWh")
            st.markdown('---')
            st.subheader("📈 Hourly Power Demand Forecast")
            st.line_chart(next_day_weather.set_index("datetime")["Predicted_Power_Demand"])
            st.markdown('---')
            st.subheader("Hourly Prediction")
            st.write(next_day_weather)
            st.markdown('---')
            report = (
                f"### ⚡️ Daily Electricity Demand Forecast {selected_city}  \n"
                    f"*Date: {datetime.now()}*  \n"
                    f"*City: {selected_city}*  \n"
                    f"This report provides the forecasted electricity demand for {selected_city} on {datetime.now()}  \n  \n"
                    f"#### 🔋 Total Predicted Demand: {adjusted_total_demand:.2f} MWh.  \n"
                    f"#### ⏰ Anticipated Peak Demand: {adjusted_peak_demand:.2f} MWh at {peak_row['datetime']}.  \n  \n"
                    f"""This forecast plays a vital role in effective grid operations by supporting optimized generation scheduling and
                    enabling proactive measures during peak demand periods. Accurate predictions contribute directly to maintaining 
                    grid stability and minimizing the risk of disruptions.  \n  \n"""
                    f"""*⚠️ Note: This forecast reflects estimated demand
                    across the entire city. More detailed regional forecasts will be provided in subsequent updates.*""")
            st.markdown(report)
