import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Main Header Styles */
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }

        /* Sub-Header Styles */
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-bottom: 1rem;
            font-weight: bold;
        }

        /* Metric Card Styles */
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }

        /* Sidebar Specific Styles */
        .css-1aumxhk {
            background-color: #2c3e50 !important;
            color: white !important;
        }

        .stSidebar .css-1aumxhk {
            background-color: #2c3e50 !important;
        }

        /* Sidebar Title */
        .css-1aumxhk h1 {
            color: #3498db !important;
            font-weight: bold;
        }

        /* Sidebar Text */
        .css-1aumxhk p, .css-1aumxhk li {
            color: #ecf0f1 !important;
        }

        /* Sidebar Selectbox */
        .stSelectbox > div > div > div {
            background-color: #34495e !important;
            color: white !important;
        }

        .stSelectbox > div > div > div:hover {
            background-color: #2980b9 !important;
        }

        /* Sidebar Info Box */
        .sidebar-info {
            background-color: #34495e;
            color: #ecf0f1;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #3498db;
        }

        /* Sidebar Multiselect */
        .stMultiSelect > div > div > div {
            background-color: #34495e !important;
            color: white !important;
        }

        .stMultiSelect > div > div > div:hover {
            background-color: #2980b9 !important;
        }

        /* Scrollbar Styles for Sidebar */
        .css-1aumxhk::-webkit-scrollbar {
            width: 10px;
        }

        .css-1aumxhk::-webkit-scrollbar-track {
            background: #34495e;
        }

        .css-1aumxhk::-webkit-scrollbar-thumb {
            background-color: #3498db;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True) 