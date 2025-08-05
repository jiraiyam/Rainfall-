import streamlit as st
from src.components.sidebar import create_sidebar
from src.components.styles import apply_custom_styles
from src.pages.home_page import render_home_page
from src.pages.data_collection_page import render_data_collection_page
from src.pages.eda_page import render_eda_page
from src.pages.forecasting_chatbot_page import render_forecasting_chatbot_page

def main():
    # Page configuration
    st.set_page_config(
        page_title="Weather Data Analysis System",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styles
    apply_custom_styles()

    # Create sidebar and get selected page
    page = create_sidebar()

    # Render the selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Data Collection":
        render_data_collection_page()
    elif page == "📈 EDA (Exploratory Data Analysis)":
        render_eda_page()
    elif page == "🔮 Forecasting & ChatBot":
        render_forecasting_chatbot_page()

if __name__ == "__main__":
    main()