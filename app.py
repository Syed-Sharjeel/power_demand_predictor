import streamlit as st
import pandas as pd
import numpy as np
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, time

st.set_page_config(page_title="Power Demand Predictor", layout="wide")

# Load cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_power_data.csv")


# Split features and target and train model
def model_fit(df):
    X = df.drop(columns=["POWER_DEMAND"])
    y = df["POWER_DEMAND"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log transform for training
    y_train_log = np.log1p(y_train)

    # Train model
    model = XGBRegressor(n_estimators=8)
    model.fit(X_train, y_train_log)
    return model


# Select City using Drop-down and get population
def city_select():
    cities_data = {
        # Dehli: {population: 22.3M, per_capita_kwh: 1075}, Pakistan 606
        "Karachi": {"population": 18868021},
        "Lahore": {"population": 13004135},
        "Islamabad": {"population": 1108872},
        "Peshawar": {"population": 1905975},
        "Quetta": {"population": 1565546},
    }
    # Dropdown for city selection
    selected_city = st.sidebar.selectbox("Select a City", cities_data.keys())
    population = cities_data[selected_city]["population"]
    return selected_city, population

# Get Latidtude and Longitude of Selected City
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
    

# Get Weather Information for Selected City
def get_weather(lat, lon):
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
    # Filter next day's data
    now = pd.Timestamp.now(tz="UTC")
    next_day = now.normalize() + pd.Timedelta(days=1)
    day_after = next_day + pd.Timedelta(days=1)
    next_day_weather = weather_df[(weather_df["datetime"] >= next_day) & (weather_df["datetime"] < day_after)].copy()
    return next_day_weather


# Making Prediction through trained model and weather data
def prediction(next_day_weather, model, population):
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
    return adjusted_total_demand, adjusted_peak_demand, peak_row


def ui(selected_city, next_day_weather, total_demand, peak_demand, peak_row):
    st.sidebar.header("🌍 Location Input")
    if st.sidebar.button("Predict Power Demand"):
        lat, lon = get_coordinates(selected_city)
    else:
        return None
    if lat and lon:
        if next_day_weather.empty:
            st.error("⚠️ Weather data for the next day is not available. Try again later.")
        else:
            st.title("⚡ Power Demand Predictor")
            st.markdown("Predicts next day's electricity consumption for a single district using weather forecast data.")
            st.markdown('---')
            st.markdown(f"🔋 **Total Predicted Demand (Next Day):** {total_demand:.2f} MWh")
            st.markdown(f"⏰ **Peak Hour:** {peak_row['datetime']} with Demand: {peak_demand:.2f} MWh")
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
                    f"#### 🔋 Total Predicted Demand: {total_demand:.2f} MWh.  \n"
                    f"#### ⏰ Anticipated Peak Demand: {peak_demand:.2f} MWh at {peak_row['datetime']}.  \n  \n"
                    f"""This forecast plays a vital role in effective grid operations by supporting optimized generation scheduling and
                    enabling proactive measures during peak demand periods. Accurate predictions contribute directly to maintaining 
                    grid stability and minimizing the risk of disruptions.  \n  \n"""
                    f"""*⚠️ Note: This forecast reflects estimated demand
                    across the entire city. More detailed regional forecasts will be provided in subsequent updates.*""")
            st.markdown(report)

def workflow():
    df = load_data()
    model = model_fit(df)
    selected_city, population = city_select()
    lat, lon = get_coordinates(selected_city)
    next_day_weather = get_weather(lat, lon)
    total_demand, peak_demand, peak_row = prediction(next_day_weather, model, population)
    ui(selected_city, next_day_weather, total_demand, peak_demand, peak_row)
if __name__ == "__main__":
    workflow()
