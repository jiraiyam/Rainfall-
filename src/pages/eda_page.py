import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose

def load_and_preprocess_data():
    """
    Load and preprocess weather data with caching
    """
    try:
        df = pd.read_csv('src/data/egypt_weather_data.csv')
        
        # Preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['City'] = df['City'].astype('category')
        
        # Compute yearly rainfall statistics
        yearly_rain = df.groupby('Year')['Rainfall_mm'].sum()
        
        return df, yearly_rain
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Wrap the function with cache_data
cached_load_and_preprocess_data = st.cache_data(load_and_preprocess_data)

def compute_city_yearly_rain(df):
    """Compute yearly rainfall for each city"""
    return df.groupby(['Year', 'City'])['Rainfall_mm'].sum().reset_index()

# Wrap with cache_data
cached_compute_city_yearly_rain = st.cache_data(compute_city_yearly_rain)

def compute_seasonal_rain(df):
    """Compute average monthly rainfall for each city"""
    return df.groupby(['City', 'Month'])['Rainfall_mm'].mean().reset_index()

# Wrap with cache_data
cached_compute_seasonal_rain = st.cache_data(compute_seasonal_rain)

def render_eda_page():
    st.markdown('<h1 class="main-header">📊 Comprehensive Rainfall Analysis</h1>', unsafe_allow_html=True)
    
    # Load and preprocess data
    df, yearly_rain = cached_load_and_preprocess_data()
    
    if df is None or yearly_rain is None:
        st.warning("⚠️ No data found! Please collect data first from the Data Collection page.")
        st.stop()
    
    # Compute key statistics
    avg_rain = yearly_rain.mean()
    max_rain = yearly_rain.max()
    min_rain = yearly_rain.min()
    max_year = yearly_rain.idxmax()
    min_year = yearly_rain.idxmin()
    
    # Create two advanced rainfall visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Plot with Advanced Styling
        fig_bar = go.Figure(data=[
            go.Bar(
                x=yearly_rain.index.astype(str),
                y=yearly_rain.values,
                marker_color=[
                    'red' if year == max_year else 
                    'green' if year == min_year else 
                    'blue' 
                    for year in yearly_rain.index
                ],
                text=[f'{val:.0f}' for val in yearly_rain.values],
                textposition='outside'
            )
        ])
        
        fig_bar.add_hline(
            y=avg_rain, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Annual Average: {avg_rain:.1f} mm",
            annotation_position="bottom right"
        )
        
        fig_bar.update_layout(
            title='Annual Rainfall Distribution',
            xaxis_title='Year',
            yaxis_title='Total Rainfall (mm)',
            height=500
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Line Plot with Area Fill
        fig_line = go.Figure(data=[
            go.Scatter(
                x=yearly_rain.index,
                y=yearly_rain.values,
                mode='lines+markers',
                line=dict(color='#2a9d8f', width=3),
                marker=dict(size=10, color='#2a9d8f'),
                fill='tozeroy',
                fillcolor='rgba(42, 157, 143, 0.2)'
            )
        ])
        
        fig_line.update_layout(
            title='Annual Rainfall Trend',
            xaxis_title='Year',
            yaxis_title='Total Rainfall (mm)',
            height=500
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # City-wise Rainfall Analysis
    st.markdown('<h2 class="sub-header">🏙️ City-wise Rainfall Comparison</h2>', unsafe_allow_html=True)
    
    # City-wise yearly rainfall
    city_yearly_rain = cached_compute_city_yearly_rain(df)
    
    # Create city-wise rainfall trend plot
    fig_city_trend = go.Figure()
    
    for city in df['City'].unique():
        city_data = city_yearly_rain[city_yearly_rain['City'] == city]
        
        fig_city_trend.add_trace(go.Scatter(
            x=city_data['Year'],
            y=city_data['Rainfall_mm'],
            mode='lines+markers',
            name=city
        ))
    
    fig_city_trend.update_layout(
        title='Yearly Rainfall Trends by City',
        xaxis_title='Year',
        yaxis_title='Total Rainfall (mm)',
        height=500,
        legend_title_text='Cities'
    )
    
    st.plotly_chart(fig_city_trend, use_container_width=True)
    
    # Seasonal Rainfall Analysis
    st.markdown('<h2 class="sub-header">Seasonal Rainfall Patterns</h2>', unsafe_allow_html=True)
    
    # Seasonal rainfall by city
    seasonal_rain = cached_compute_seasonal_rain(df)
    
    # Seasonal rainfall plot
    fig_seasonal = go.Figure()
    
    for city in df['City'].unique():
        city_data = seasonal_rain[seasonal_rain['City'] == city]
        
        fig_seasonal.add_trace(go.Scatter(
            x=city_data['Month'],
            y=city_data['Rainfall_mm'],
            mode='lines+markers',
            name=city
        ))
    
    fig_seasonal.update_layout(
        title='Average Monthly Rainfall by City',
        xaxis_title='Month',
        yaxis_title='Average Rainfall (mm)',
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
    )
    
    st.plotly_chart(fig_seasonal, use_container_width=True) 

    # New Sections: Trend Decomposition and Drought Analysis
    st.markdown('<h2 class="sub-header">🔬 Advanced Rainfall Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Trend Decomposition
    st.markdown('### 📈 Seasonal Trend Decomposition')
    
    # Create a figure with subplots for each city
    fig_decomp = make_subplots(
        rows=len(df['City'].unique()), 
        cols=1, 
        subplot_titles=[f"{city} - Rainfall Trend Component" for city in df['City'].unique()],
        vertical_spacing=0.1
    )
    
    for i, city in enumerate(df['City'].unique(), 1):
        # Monthly time series for the city
        city_df = df[df['City'] == city].copy()
        monthly = city_df.set_index('Date').resample('MS')['Rainfall_mm'].sum().reset_index()
        monthly.set_index('Date', inplace=True)
        
        # Interpolate any missing values
        if monthly['Rainfall_mm'].isnull().sum() > 0:
            monthly['Rainfall_mm'] = monthly['Rainfall_mm'].interpolate()
        
        try:
            # Seasonal decomposition
            decomposition = seasonal_decompose(monthly['Rainfall_mm'], model='additive', period=12)
            
            # Add trend to subplot
            fig_decomp.add_trace(
                go.Scatter(
                    x=decomposition.trend.index, 
                    y=decomposition.trend.values,
                    mode='lines',
                    name=f'{city} Trend',
                    line=dict(color='#3498db', width=2.5)
                ),
                row=i, col=1
            )
        except Exception as e:
            st.error(f"Error in trend decomposition for {city}: {e}")
    
    fig_decomp.update_layout(
        height=1000,
        title_text="Seasonal Trend Decomposition for Rainfall"
    )
    
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    # Drought Analysis
    st.markdown('### 🏜️ Drought Detection Analysis')
    
    # Rolling 30-day rainfall per city
    df['Rolling_30d'] = df.groupby('City', observed=True)['Rainfall_mm'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    
    # Define drought: less than 1 mm average rainfall over 30 days
    df['Drought'] = df.groupby('City', observed=True)['Rolling_30d'].transform(lambda x: x < 1)
    
    # Drought Analysis Visualization
    fig_drought = make_subplots(
        rows=len(df['City'].unique()), 
        cols=1, 
        subplot_titles=[f"30-Day Rolling Rainfall & Droughts in {city}" for city in df['City'].unique()],
        vertical_spacing=0.1
    )
    
    for i, city in enumerate(df['City'].unique(), 1):
        city_df = df[df['City'] == city].copy()
        
        # Rolling rainfall trace
        fig_drought.add_trace(
            go.Scatter(
                x=city_df['Date'], 
                y=city_df['Rolling_30d'],
                mode='lines',
                name=f'{city} Rolling Rainfall',
                line=dict(color='#2c3e50', width=2)
            ),
            row=i, col=1
        )
        
        # Drought periods
        drought_periods = city_df[city_df['Drought']]
        if not drought_periods.empty:
            fig_drought.add_trace(
                go.Scatter(
                    x=drought_periods['Date'], 
                    y=drought_periods['Rolling_30d'],
                    mode='markers',
                    name=f'{city} Drought Periods',
                    marker=dict(color='#e74c3c', size=8, opacity=0.6)
                ),
                row=i, col=1
            )
    
    fig_drought.update_layout(
        height=300 * len(df['City'].unique()),
        title='Rainfall Trends & Drought Detection per City',
        showlegend=False
    )
    
    st.plotly_chart(fig_drought, use_container_width=True)
    
    # Advanced Insights Report with Enhanced Visualization
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .insights-container {
        background: linear-gradient(145deg, #f9fafb 0%, #f0f3f6 100%);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #3498db;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .insights-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .insights-header {
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.5em;
        border-bottom: 3px solid #3498db;
        padding-bottom: 15px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    
    .insights-header i {
        margin-right: 15px;
        color: #3498db;
    }
    
    .insights-item {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .insights-item:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        background-color: #f7f9fc;
    }
    
    .city-name {
        color: #2980b9;
        font-weight: 700;
        font-size: 1.1em;
        flex-grow: 1;
    }
    
    .trend-significant {
        color: #27ae60;
        font-weight: 600;
        background-color: rgba(39, 174, 96, 0.1);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .trend-not-significant {
        color: #c0392b;
        font-weight: 600;
        background-color: rgba(192, 57, 43, 0.1);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .drought-extreme {
        color: #c0392b;
        font-weight: 700;
        background-color: rgba(192, 57, 43, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .drought-severe {
        color: #d35400;
        font-weight: 700;
        background-color: rgba(211, 84, 0, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .drought-moderate {
        color: #f39c12;
        font-weight: 700;
        background-color: rgba(243, 156, 18, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .drought-low {
        color: #27ae60;
        font-weight: 700;
        background-color: rgba(39, 174, 96, 0.15);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    .insights-detail {
        color: #7f8c8d;
        font-size: 0.9em;
        margin-left: 15px;
    }
    
    .conclusion {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        border-radius: 15px;
        padding: 25px;
        margin-top: 25px;
        position: relative;
        overflow: hidden;
    }
    
    .conclusion::before {
        content: '🌍';
        position: absolute;
        font-size: 5em;
        opacity: 0.1;
        right: -20px;
        top: -20px;
    }
    
    .conclusion-title {
        color: #3498db;
        font-weight: 700;
        font-size: 1.5em;
        margin-bottom: 20px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    .conclusion-point {
        margin-bottom: 15px;
        position: relative;
        padding-left: 30px;
        font-weight: 300;
        line-height: 1.6;
    }
    
    .conclusion-point::before {
        content: '✓';
        color: #3498db;
        position: absolute;
        left: 0;
        top: 0;
        font-weight: bold;
    }
    
    .data-highlight {
        display: inline-block;
        background-color: rgba(52, 152, 219, 0.1);
        color: #2980b9;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 5px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Long-Term Rainfall Trends
    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
    st.markdown('<div class="insights-header"><i class="fas fa-chart-line"></i>1. Long-Term Rainfall Trends</div>', unsafe_allow_html=True)
    
    trends = [
        {"city": "Cairo", "trend": "increasing", "rate": 1.36, "r2": 0.34, "significance": "significant"},
        {"city": "Alexandria", "trend": "increasing", "rate": 6.49, "r2": 0.58, "significance": "significant"},
        {"city": "Aswan", "trend": "increasing", "rate": 0.12, "r2": 0.10, "significance": "not significant"},
        {"city": "Luxor", "trend": "increasing", "rate": 0.11, "r2": 0.03, "significance": "not significant"},
        {"city": "Mansoura", "trend": "increasing", "rate": 1.78, "r2": 0.28, "significance": "significant"},
        {"city": "Tanta", "trend": "increasing", "rate": 1.63, "r2": 0.35, "significance": "significant"}
    ]
    
    for trend in trends:
        st.markdown(f"""
        <div class="insights-item">
            <div class="city-name">{trend['city']}</div>
            <div class="insights-detail">
                {trend['trend']} trend 
                (<span class="data-highlight">{trend['rate']:.2f} mm/year</span>, 
                R²=<span class="data-highlight">{trend['r2']:.2f}</span>) - 
                <span class="trend-{'significant' if trend['significance'] == 'significant' else 'not-significant'}">
                    {trend['significance']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>')
    
    # Seasonal Rainfall Patterns
    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
    st.markdown('<div class="insights-header"><i class="fas fa-cloud-sun"></i>2. Seasonal Rainfall Patterns</div>', unsafe_allow_html=True)
    
    seasonal_patterns = [
        {"city": "Cairo", "wettest": "Winter", "wettest_mm": 305.1, "driest": "Summer", "driest_mm": 0.9},
        {"city": "Alexandria", "wettest": "Winter", "wettest_mm": 1381.6, "driest": "Summer", "driest_mm": 24.5},
        {"city": "Aswan", "wettest": "Winter", "wettest_mm": 31.5, "driest": "Summer", "driest_mm": 0.4},
        {"city": "Luxor", "wettest": "Winter", "wettest_mm": 37.0, "driest": "Summer", "driest_mm": 0.0},
        {"city": "Mansoura", "wettest": "Winter", "wettest_mm": 553.3, "driest": "Summer", "driest_mm": 7.6},
        {"city": "Tanta", "wettest": "Winter", "wettest_mm": 486.4, "driest": "Summer", "driest_mm": 5.1}
    ]
    
    for pattern in seasonal_patterns:
        st.markdown(f"""
        <div class="insights-item">
            <div class="city-name">{pattern['city']}</div>
            <div class="insights-detail">
                Wettest: <span class="data-highlight">{pattern['wettest']} ({pattern['wettest_mm']:.1f}mm)</span>, 
                Driest: <span class="data-highlight">{pattern['driest']} ({pattern['driest_mm']:.1f}mm)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>')
    
    # Drought Analysis
    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
    st.markdown('<div class="insights-header"><i class="fas fa-water"></i>3. Drought Analysis</div>', unsafe_allow_html=True)
    
    drought_analysis = [
        {"city": "Cairo", "days": 8736, "percentage": 99.7},
        {"city": "Alexandria", "days": 8010, "percentage": 91.4},
        {"city": "Aswan", "days": 8766, "percentage": 100.0},
        {"city": "Luxor", "days": 8766, "percentage": 100.0},
        {"city": "Mansoura", "days": 8679, "percentage": 99.0},
        {"city": "Tanta", "days": 8711, "percentage": 99.4}
    ]
    
    for drought in drought_analysis:
        # Determine drought severity
        if drought['percentage'] == 100.0:
            severity_class = 'drought-extreme'
            severity_text = 'Extreme'
        elif drought['percentage'] >= 95.0:
            severity_class = 'drought-severe'
            severity_text = 'Severe'
        elif drought['percentage'] >= 90.0:
            severity_class = 'drought-moderate'
            severity_text = 'Moderate'
        else:
            severity_class = 'drought-low'
            severity_text = 'Low'
        
        st.markdown(f"""
        <div class="insights-item">
            <div class="city-name">{drought['city']}</div>
            <div class="insights-detail">
                <span class="data-highlight">{drought['days']} drought days</span>
                (<span class="{severity_class}">
                    {severity_text}: {drought['percentage']:.1f}% of total days
                </span>)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>')
    
    # Conclusion Section
    st.markdown("""
    <div class="conclusion">
        <div class="conclusion-title">🌍 Conclusion: Urban Rainfall Patterns</div>
        <div class="conclusion-content">
            <div class="conclusion-point">Climate variability is evident across Egyptian cities</div>
            <div class="conclusion-point">Trend analysis reveals complex rainfall dynamics</div>
            <div class="conclusion-point">Drought periods highlight critical water resource challenges</div>
            <div class="conclusion-point">Continued monitoring is crucial for understanding long-term climate trends</div>
        </div>
    </div>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    """, unsafe_allow_html=True) 