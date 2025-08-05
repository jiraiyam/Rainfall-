import streamlit as st
from datetime import datetime
from src.utils.api_handler import WeatherDataCollector
import pandas as pd

def render_data_collection_page():
    st.markdown('<h1 class="main-header">📊 Weather Data Collection</h1>', unsafe_allow_html=True)
    
    # Configuration section
    st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Date Range**")
        start_date = st.date_input("Start Date", value=datetime(2000, 7, 1))
        end_date = st.date_input("End Date", value=datetime(2024, 6, 30))
    
    with col2:
        st.markdown("**🏙️ Cities Selection**")
        data_collector = WeatherDataCollector()
        
        selected_cities = st.multiselect(
            "Select cities to collect data for:",
            list(data_collector.egyptian_cities.keys()),
            default=list(data_collector.egyptian_cities.keys())
        )
    
    # Parameters section
    st.markdown("**🌤️ Weather Parameters**")
    
    param_descriptions = {
        "precipitation_sum": "Daily precipitation sum (mm)",
        "temperature_2m_max": "Maximum daily temperature at 2m (°C)",
        "temperature_2m_min": "Minimum daily temperature at 2m (°C)",
        "windspeed_10m_max": "Maximum daily wind speed at 10m (km/h)"
    }
    
    for param in data_collector.weather_vars:
        st.markdown(f"✅ **{param}**: {param_descriptions[param]}")
    
    st.markdown("---")
    
    # Data collection button and status
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Data Collection", type="primary", use_container_width=True):
            if not selected_cities:
                st.error("Please select at least one city!")
            else:
                # Collect data
                city_dataframes = data_collector.collect_data(start_date, end_date, selected_cities)
                
                if city_dataframes:
                    # Save to session state
                    st.session_state['weather_data'] = city_dataframes
                    
                    # Save to CSV
                    data_collector.save_data(city_dataframes)
                    
                    # Display summary statistics
                    st.markdown("---")
                    st.markdown('<h2 class="sub-header">📈 Collection Summary</h2>', unsafe_allow_html=True)
                    
                    # Overall Collection Summary
                    st.markdown("### 🌍 Overall Collection Overview")
                    
                    # Prepare overall summary metrics
                    total_records = sum(len(df) for df in city_dataframes.values())
                    total_cities = len(city_dataframes)
                    earliest_date = min(df['Date'].min() for df in city_dataframes.values())
                    latest_date = max(df['Date'].max() for df in city_dataframes.values())
                    
                    # Create columns for overall summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🏙️ Total Cities", total_cities)
                    with col2:
                        st.metric("📊 Total Records", total_records)
                    with col3:
                        st.metric("📅 Start Date", earliest_date.strftime('%Y-%m-%d'))
                    with col4:
                        st.metric("📅 End Date", latest_date.strftime('%Y-%m-%d'))
                    
                    # Detailed City-wise Summary
                    st.markdown("### 🌆 City-wise Breakdown")
                    
                    # Create a summary table
                    summary_data = []
                    for city, df in city_dataframes.items():
                        summary_data.append({
                            'City': city,
                            'Records': len(df),
                            'Start Date': df['Date'].min().strftime('%Y-%m-%d'),
                            'End Date': df['Date'].max().strftime('%Y-%m-%d'),
                            'Avg Max Temp (°C)': round(df['Temp_Max_C'].mean(), 2),
                            'Avg Min Temp (°C)': round(df['Temp_Min_C'].mean(), 2),
                            'Total Rainfall (mm)': round(df['Rainfall_mm'].sum(), 2),
                            'Max Wind Speed (km/h)': round(df['WindSpeed_Max_kmh'].max(), 2)
                        })
                    
                    # Display summary table
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visualizations
                    st.markdown("### 📊 Data Visualization")
                    
                    # Prepare data for visualization
                    import plotly.express as px
                    import plotly.graph_objs as go
                    
                    # Temperature Comparison
                    temp_data = []
                    for city, df in city_dataframes.items():
                        temp_data.append(go.Box(
                            y=df['Temp_Max_C'],
                            name=f'{city} Max Temp',
                            boxpoints='outliers'
                        ))
                        temp_data.append(go.Box(
                            y=df['Temp_Min_C'],
                            name=f'{city} Min Temp',
                            boxpoints='outliers'
                        ))
                    
                    fig_temp = go.Figure(data=temp_data)
                    fig_temp.update_layout(
                        title='Temperature Distribution by City',
                        yaxis_title='Temperature (°C)',
                        boxmode='group'
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                    
                    # Rainfall Comparison
                    rainfall_data = []
                    for city, df in city_dataframes.items():
                        rainfall_data.append(go.Bar(
                            x=[city],
                            y=[df['Rainfall_mm'].sum()],
                            name=city
                        ))
                    
                    fig_rainfall = go.Figure(data=rainfall_data)
                    fig_rainfall.update_layout(
                        title='Total Rainfall by City',
                        yaxis_title='Total Rainfall (mm)',
                        xaxis_title='City'
                    )
                    st.plotly_chart(fig_rainfall, use_container_width=True)
                    
                else:
                    st.error("❌ No data was collected. Please check your connection and try again.")
    
    st.markdown("---")
    
    # Information section
    st.markdown('<h2 class="sub-header">ℹ️ Data Collection Information</h2>', unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **📡 Data Source**: Open-Meteo Archive API
        
        **🔄 Update Frequency**: Historical data (2000-2024)
        
        **📊 Data Quality**: 
        - Automatic retry mechanism for failed requests
        - Rate limiting protection
        - Data validation and cleaning
        
        **🌍 Geographic Coverage**:
        - 6 major Egyptian cities
        - Diverse climate zones
        - Strategic locations across Egypt
        """)
    
    with info_col2:
        st.markdown("""
        **⚡ Technical Details**:
        - REST API integration
        - JSON data format
        - Daily resolution
        - UTC+2 timezone (Africa/Cairo)
        
        **🛡️ Error Handling**:
        - Automatic retry on failures
        - Rate limit management
        - Connection timeout handling
        - Data integrity checks
        """)
    
    # Show existing data if available
    if 'weather_data' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">💾 Currently Loaded Data</h2>', unsafe_allow_html=True)
        
        city_dataframes = st.session_state['weather_data']
        
        # Create columns for each city
        cols = st.columns(len(city_dataframes))
        
        for i, (city, df) in enumerate(city_dataframes.items()):
            with cols[i]:
                st.markdown(f"**{city} Data**")
                st.metric("Records", len(df))
                st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
                st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d')) 