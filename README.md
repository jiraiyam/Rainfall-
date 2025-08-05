# Weather Data Analysis System 🌤️

## Overview
A comprehensive weather data analysis system for Egyptian cities from 2000-2024, providing interactive visualizations, statistical insights, and forecasting capabilities.

## Features
- 📊 Data Collection from Open-Meteo API
- 📈 Exploratory Data Analysis (EDA)
- 🔮 Weather Forecasting
- 🤖 Intelligent Weather ChatBot

## Installation

### Prerequisites
- Python 3.9+
- pip

### Steps
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## Project Structure
```
RainFull/
│
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
│
├── src/
│   ├── pages/             # Individual page components
│   │   ├── home.py
│   │   ├── data_collection.py
│   │   ├── eda.py
│   │   └── forecasting.py
│   │
│   ├── components/        # Reusable UI components
│   │   ├── sidebar.py
│   │   └── styles.py
│   │
│   └── utils/             # Utility functions
│       ├── data_processing.py
│       └── chatbot.py
│
└── data/                 # Data storage
    └── egypt_weather_2000_2024.csv
```

## Data Sources
- Open-Meteo Historical Weather API
- Covers 6 major Egyptian cities
- Daily resolution from 2000-2024

## Technologies
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Statsmodels

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify your license here]

## Contact
[Your contact information] 