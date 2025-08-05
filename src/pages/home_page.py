import streamlit as st

def render_home_page():
    st.markdown('<h1 class="main-header">🌤️ Weather Data Analysis System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-subtitle">
    Comprehensive Weather Analysis for Egyptian Cities (2000-2024)
    </div>
    """, unsafe_allow_html=True)
    
    # Project overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card data-collection">
        <h3>📊 Data Collection</h3>
        <p>Automated weather data collection from Open-Meteo API for 6 major Egyptian cities covering 24 years of historical data.</p>
        <ul>
        <li>Temperature (Max/Min)</li>
        <li>Rainfall</li>
        <li>Wind Speed</li>
        <li>Daily Resolution</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card data-analysis">
        <h3>📈 Data Analysis</h3>
        <p>Interactive exploratory data analysis with comprehensive visualizations and statistical insights.</p>
        <ul>
        <li>Trend Analysis</li>
        <li>Seasonal Patterns</li>
        <li>Drought Detection</li>
        <li>City Comparisons</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card forecasting">
        <h3>🔮 Forecasting</h3>
        <p>Advanced weather forecasting models and intelligent chatbot for weather insights and predictions.</p>
        <ul>
        <li>Time Series Forecasting</li>
        <li>Interactive ChatBot</li>
        <li>Predictive Analytics</li>
        <li>Real-time Insights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Custom CSS for home page
    st.markdown("""
    <style>
        /* Home Page Subtitle */
        .home-subtitle {
            text-align: center; 
            font-size: 1.2rem; 
            color: #34495e; 
            margin-bottom: 2rem;
            font-weight: 500;
        }

        /* Feature Cards */
        .feature-card {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-left: 5px solid #3498db;
        }

        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }

        .feature-card h3 {
            color: #3498db;
            margin-bottom: 1rem;
            font-weight: bold;
        }

        .feature-card p {
            margin-bottom: 1rem;
            line-height: 1.6;
        }

        .feature-card ul {
            list-style-type: none;
            padding-left: 0;
        }

        .feature-card ul li {
            padding: 0.3rem 0;
            position: relative;
            padding-left: 20px;
        }

        .feature-card ul li:before {
            content: '•';
            color: #3498db;
            font-weight: bold;
            position: absolute;
            left: 0;
        }

        /* Cities Section */
        .cities-section {
            background-color: #34495e;
            color: #ecf0f1;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }

        .cities-section h2 {
            color: #3498db;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cities covered
    st.markdown('<h2 class="sub-header" style="color: #2c3e50;">🏙️ Cities Covered</h2>', unsafe_allow_html=True)
    
    cities_info = {
        "Cairo": {"lat": 30.0444, "lon": 31.2357, "desc": "Capital and largest city"},
        "Alexandria": {"lat": 31.2001, "lon": 29.9187, "desc": "Mediterranean coastal city"},
        "Aswan": {"lat": 24.0889, "lon": 32.8998, "desc": "Southern Egypt, near Nile"},
        "Luxor": {"lat": 25.6872, "lon": 32.6396, "desc": "Historic city, Valley of Kings"},
        "Mansoura": {"lat": 31.0364, "lon": 31.3807, "desc": "Nile Delta region"},
        "Tanta": {"lat": 30.7865, "lon": 30.9982, "desc": "Agricultural center"}
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="cities-section">', unsafe_allow_html=True)
        for i, (city, info) in enumerate(list(cities_info.items())[:3]):
            st.markdown(f"""
            **{city}** ({info['lat']:.2f}, {info['lon']:.2f})  
            *{info['desc']}*
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="cities-section">', unsafe_allow_html=True)
        for city, info in list(cities_info.items())[3:]:
            st.markdown(f"""
            **{city}** ({info['lat']:.2f}, {info['lon']:.2f})  
            *{info['desc']}*
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key features
    st.markdown('<h2 class="sub-header" style="color: #2c3e50;">🔥 Key Features</h2>', unsafe_allow_html=True)
    
    features = [
        "📥 **Automated Data Collection**: Real-time API integration with error handling and retry mechanisms",
        "📊 **Interactive Visualizations**: Dynamic charts and graphs using Plotly and Matplotlib",
        "🔍 **Statistical Analysis**: Comprehensive EDA with trend analysis and pattern recognition",
        "🌡️ **Multi-Parameter Analysis**: Temperature, rainfall, wind speed correlations",
        "📅 **Time Series Analysis**: Seasonal decomposition and trend identification",
        "🏜️ **Drought Detection**: Advanced algorithms for identifying drought periods",
        "🤖 **AI ChatBot**: Intelligent assistant for weather insights and queries",
        "📈 **Forecasting Models**: Predictive analytics for future weather patterns"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.markdown("---")
    
    # Getting started
    st.markdown('<h2 class="sub-header" style="color: #2c3e50;">🚀 Getting Started</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Data Collection**: Start by collecting fresh weather data from the API
    2. **Explore Data**: Use the EDA page to understand patterns and trends
    3. **Forecasting**: Generate predictions and interact with the ChatBot
    
    Use the sidebar navigation to move between different sections of the application.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #34495e; margin-top: 2rem;">
    <p>Weather Data Analysis System | Built with Streamlit | Data from Open-Meteo API</p>
    <p>© 2024 | Graduation Project</p>
    </div>
    """, unsafe_allow_html=True) 