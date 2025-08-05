import requests
import pandas as pd
import time
import streamlit as st

class WeatherDataCollector:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.egyptian_cities = {
            "Cairo": (30.0444, 31.2357),
            "Alexandria": (31.2001, 29.9187),
            "Aswan": (24.0889, 32.8998),
            "Luxor": (25.6872, 32.6396),
            "Mansoura": (31.0364, 31.3807),
            "Tanta": (30.7865, 30.9982)
        }
        self.weather_vars = [
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "windspeed_10m_max"
        ]

    def collect_data(self, start_date, end_date, selected_cities=None):
        """
        Collect weather data for specified cities and date range
        
        Args:
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            selected_cities (list, optional): List of cities to collect data for
        
        Returns:
            dict: Dictionary of DataFrames with city names as keys
        """
        import streamlit as st
        
        if selected_cities is None:
            selected_cities = list(self.egyptian_cities.keys())

        city_dataframes = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, city in enumerate(selected_cities):
            lat, lon = self.egyptian_cities[city]
            status_text.text(f"🔄 Fetching data for {city}...")
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": ",".join(self.weather_vars),
                "timezone": "Africa/Cairo"
            }
            
            success = False
            retries = 0
            
            while not success and retries < 3:
                try:
                    response = requests.get(self.base_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        daily_data = data["daily"]
                        
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(daily_data["time"]),
                            "Rainfall_mm": daily_data["precipitation_sum"],
                            "Temp_Max_C": daily_data["temperature_2m_max"],
                            "Temp_Min_C": daily_data["temperature_2m_min"],
                            "WindSpeed_Max_kmh": daily_data["windspeed_10m_max"],
                            "City": city
                        })
                        
                        city_dataframes[city] = df
                        st.success(f"✅ Successfully collected data for {city}")
                        success = True
                        
                    elif response.status_code == 429:
                        retries += 1
                        st.warning(f"⏳ Rate limit hit. Retrying {city} (Retry {retries}/3)...")
                        time.sleep(30)
                    else:
                        st.error(f"❌ Error fetching {city}: {response.status_code}")
                        break
                        
                except Exception as e:
                    st.error(f"❌ Exception for {city}: {e}")
                    break
            
            # Update progress
            progress_bar.progress((i + 1) / len(selected_cities))
            time.sleep(2)  # Brief delay between requests
        
        # Return dictionary of DataFrames
        return city_dataframes

    def save_data(self, city_dataframes, filename='egypt_weather_data.csv'):
        """
        Save collected data to CSV
        
        Args:
            city_dataframes (dict): Dictionary of DataFrames with city names as keys
            filename (str, optional): Output filename template
        """
        import streamlit as st
        
        for city, df in city_dataframes.items():
            # Create a city-specific filename
            city_filename = f'{city.lower()}_weather_data.csv'
            df.to_csv(f'src/data/{city_filename}', index=False)
        
        st.success(f"📥 Data saved for {len(city_dataframes)} cities")

    def load_data(self, filename='egypt_weather_data.csv'):
        """
        Load weather data from CSV
        
        Args:
            filename (str, optional): Input filename
        
        Returns:
            pd.DataFrame: Loaded weather data
        """
        try:
            df = pd.read_csv(f'src/data/{filename}')
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except FileNotFoundError:
            st.error(f"❌ File {filename} not found!")
            return None 