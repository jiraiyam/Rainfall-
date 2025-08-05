import streamlit as st

def create_sidebar():
    st.sidebar.title("🌤️ Weather Analysis System")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Collection", "📈 EDA (Exploratory Data Analysis)", "🔮 Forecasting & ChatBot"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="sidebar-info">
    <h4>About This Project</h4>
    <p>A comprehensive weather data analysis system for Egyptian cities from 2000-2024.</p>
    <ul>
    <li>Real-time data collection</li>
    <li>Interactive visualizations</li>
    <li>Statistical analysis</li>
    <li>Weather forecasting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    return page 