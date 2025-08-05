import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import plotly.graph_objs as go
import plotly.express as px
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv, dotenv_values
import time
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Add these imports at the top of the file
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import os
import time
import re

# Predefined weather and rainfall-related keywords
WEATHER_KEYWORDS = [
    'rainfall', 'rain', 'precipitation', 'weather', 'climate', 'storm', 
    'forecast', 'meteorology', 'humidity', 'temperature', 'wind', 
    'precipitation patterns', 'drought', 'flood', 'monsoon', 
    'seasonal rainfall', 'weather prediction', 'climate change'
]

def is_weather_related_query(query):
    """
    Check if the query is related to weather or rainfall
    
    Args:
        query (str): User's input query
    
    Returns:
        bool: True if query is weather-related, False otherwise
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for weather keywords
    if any(keyword in query_lower for keyword in WEATHER_KEYWORDS):
        return True
    
    return False

def get_system_prompt():
    """
    Generate a comprehensive system prompt for rainfall-specific interactions
    
    Returns:
        str: Detailed system prompt for the chatbot
    """
    return """You are an advanced Rainfall Insights AI Assistant specialized in weather and precipitation analysis. 
Your primary focus is on providing expert insights about rainfall, weather patterns, and climate-related information. 
Key capabilities:
- Explain rainfall prediction methodologies
- Analyze precipitation patterns
- Discuss climate change impacts on rainfall
- Provide scientific explanations about meteorological phenomena
- Help users understand weather data and forecasting techniques

Guidelines:
1. Always relate your responses to rainfall or weather
2. Use scientific and technical language appropriate for meteorological discussions
3. If a query is not related to weather, politely explain that you can only discuss rainfall and weather topics
4. Provide clear, concise, and informative answers
5. Use data-driven explanations when possible
6. Be prepared to break down complex meteorological concepts

Preferred response style:
- Technical yet accessible language
- Include relevant scientific context
- Cite meteorological principles when explaining phenomena"""

def get_enhanced_system_prompt():
    """
    Generate an enhanced system prompt focused on rainfall and its impacts
    
    Returns:
        str: Detailed system prompt for rainfall impact analysis
    """
    return """You are a specialized Rainfall Impact Analysis AI Assistant, an expert in precipitation patterns and their comprehensive impacts on society, environment, and economy. 

Your core expertise includes:

🌧️ **RAINFALL ANALYSIS:**
- Precipitation pattern analysis and trends
- Seasonal and annual rainfall variations
- Extreme rainfall events (droughts, floods, heavy precipitation)
- Rainfall forecasting and prediction methodologies
- Climate change impacts on precipitation patterns

🌍 **ENVIRONMENTAL IMPACTS:**
- Soil erosion and land degradation effects
- Water resource management and availability
- Ecosystem health and biodiversity impacts
- Groundwater recharge and depletion
- River flow patterns and flood risks
- Desertification and land use changes

🌾 **AGRICULTURAL IMPACTS:**
- Crop yield variations due to rainfall patterns
- Irrigation needs and water stress on agriculture
- Drought impacts on farming communities
- Flood damage to agricultural lands
- Seasonal planting and harvesting considerations
- Food security implications

💰 **ECONOMIC IMPACTS:**
- Agricultural economic losses from rainfall extremes
- Infrastructure damage from floods and droughts
- Water supply costs and management expenses
- Insurance claims related to weather events
- Tourism and outdoor activity impacts
- Energy sector implications (hydropower, cooling needs)

👥 **SOCIAL IMPACTS:**
- Community displacement due to extreme rainfall events
- Public health implications (water quality, disease spread)
- Urban planning and flood management needs
- Rural vs urban rainfall impact disparities
- Emergency response and disaster preparedness

🏙️ **URBAN IMPACTS:**
- Urban flooding and drainage system capacity
- Heat island effects and rainfall patterns
- Infrastructure resilience to extreme weather
- Water supply and demand management
- Stormwater management systems

**RESPONSE GUIDELINES:**
1. Always relate responses to rainfall and its direct/indirect impacts
2. Provide scientific explanations with practical implications
3. Use data-driven insights when weather data is available
4. Explain both immediate and long-term consequences
5. Consider multiple impact dimensions (environmental, social, economic)
6. Offer actionable insights and recommendations when appropriate
7. Use clear, accessible language while maintaining scientific accuracy

**PREFERRED RESPONSE STYLE:**
- Start with direct rainfall analysis
- Explain the impact mechanisms
- Provide specific examples from Egyptian context when relevant
- Include quantitative insights from available data
- Suggest mitigation or adaptation strategies when appropriate
- Use emojis and formatting for better readability

Remember: You are the expert on how rainfall affects everything - from individual farmers to entire cities, from ecosystems to economies."""

def render_ai_chatbot():
    """
    Enhanced AI Chatbot for rainfall impact analysis with simple, clean interface
    """
    st.markdown("## 🤖 Rainfall Insights AI Chatbot")
    st.markdown("*Specialized AI assistant for analyzing rainfall patterns and their comprehensive impacts*")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "weather_data" not in st.session_state:
        st.session_state.weather_data = load_weather_data_context()
    
    # Load weather data context
    weather_data = st.session_state.weather_data
    
    # Display data context in an expandable section
    if weather_data is not None and not weather_data.empty:
        with st.expander("📊 Current Weather Data Context", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            total_records = len(weather_data)
            try:
                date_range = f"{weather_data['Date'].min().strftime('%Y-%m-%d')} to {weather_data['Date'].max().strftime('%Y-%m-%d')}"
            except:
                date_range = "Date range unavailable"
            
            cities = weather_data['City'].unique() if 'City' in weather_data.columns else ['N/A']
            cities_str = f"{len(cities)} cities" if len(cities) > 3 else ", ".join(cities)
            
            with col1:
                st.metric("Total Records", f"{total_records:,}")
            with col2:
                st.metric("Date Range", date_range)
            with col3:
                st.metric("Cities", cities_str)
            with col4:
                if 'Rainfall' in weather_data.columns:
                    avg_rainfall = f"{weather_data['Rainfall'].mean():.1f}mm"
                else:
                    avg_rainfall = "N/A"
                st.metric("Avg Rainfall", avg_rainfall)
    
    # Quick action buttons for common rainfall impact queries
    st.markdown("### 🚀 Quick Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_queries = [
        ("🌾 Agricultural Impact", "How does rainfall affect crop yields and farming in Egypt?"),
        ("🏙️ Urban Flooding", "What are the urban flooding risks from heavy rainfall?"),
        ("💧 Water Resources", "How does rainfall impact water availability and management?"),
        ("🌍 Environmental Effects", "What are the environmental consequences of rainfall patterns?"),
        ("💰 Economic Impact", "How does rainfall affect the Egyptian economy?"),
        ("🏥 Health & Social", "What are the health and social impacts of rainfall changes?"),
        ("⚡ Extreme Events", "How do extreme rainfall events affect communities?"),
        ("🔮 Future Projections", "What are the future rainfall trends and implications?")
    ]
    
    for i, (label, query) in enumerate(quick_queries):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            if st.button(label, key=f"quick_{i}", help=f"Ask: {query}"):
                st.session_state.quick_query = query
    
    # Handle quick query
    if hasattr(st.session_state, 'quick_query'):
        prompt = st.session_state.quick_query
        delattr(st.session_state, 'quick_query')
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate AI response
        with st.spinner("🤔 Analyzing rainfall impacts..."):
            try:
                # Prepare enhanced context
                enhanced_messages = prepare_enhanced_context(
                    st.session_state.messages, weather_data, prompt
                )
                
                # Get AI response
                response = get_ai_response(enhanced_messages)
                
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Generate visualization if relevant
                    chart = generate_rainfall_visualization(prompt, weather_data)
                    if chart:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "Here's a relevant visualization:",
                            "chart": chart
                        })
                else:
                    st.error("Failed to get AI response. Please try again.")
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display chart if present
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about rainfall impacts, patterns, or consequences..."):
        if is_weather_related_query(prompt):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing rainfall impacts..."):
                    try:
                        # Prepare enhanced context
                        enhanced_messages = prepare_enhanced_context(
                            st.session_state.messages, weather_data, prompt
                        )
                        
                        # Get AI response
                        response = get_ai_response(enhanced_messages)
                        
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Generate visualization if relevant
                            chart = generate_rainfall_visualization(prompt, weather_data)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": "Here's a relevant visualization:",
                                    "chart": chart
                                })
                        else:
                            st.error("Failed to get AI response. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
        else:
            st.warning("⚠️ Please ask questions related to rainfall, weather patterns, or their impacts. This chatbot specializes in rainfall impact analysis.")
    
    # Smart suggestions based on conversation context
    if st.session_state.messages:
        suggestions = generate_smart_suggestions(st.session_state.messages, weather_data)
        if suggestions:
            st.markdown("### 💡 Suggested Follow-up Questions")
            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
                if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.quick_query = suggestion

def load_weather_data_context():
    """
    Load weather data context from the actual CSV files
    
    Returns:
        pd.DataFrame: Weather data context or None if loading fails
    """
    try:
        # Try to load the main Egypt weather data file
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'egypt_weather_2000_2024.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Ensure Date column is datetime - Fix the strftime error
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            elif 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop('date', axis=1)  # Remove the old column
                
            # Remove any rows with invalid dates
            df = df.dropna(subset=['Date'])
                
            # Limit to recent data for better performance (last 2 years)
            if 'Date' in df.columns and not df.empty:
                try:
                    recent_date = df['Date'].max() - pd.DateOffset(years=2)
                    df = df[df['Date'] >= recent_date]
                except:
                    # If date operations fail, just use the data as is
                    pass
            
            # Limit rows for performance (max 10000 records)
            if len(df) > 10000:
                df = df.tail(10000)
                
            return df
        else:
            # Fallback: try to load individual city files
            city_files = [
                'cairo_weather_data.csv',
                'alexandria_weather_data.csv',
                'aswan_weather_data.csv'
            ]
            
            dfs = []
            for city_file in city_files:
                city_path = os.path.join(os.path.dirname(__file__), '..', 'data', city_file)
                if os.path.exists(city_path):
                    city_df = pd.read_csv(city_path)
                    
                    # Fix datetime handling
                    if 'Date' in city_df.columns:
                        city_df['Date'] = pd.to_datetime(city_df['Date'], errors='coerce')
                    elif 'date' in city_df.columns:
                        city_df['Date'] = pd.to_datetime(city_df['date'], errors='coerce')
                        city_df = city_df.drop('date', axis=1)
                    
                    # Remove rows with invalid dates
                    city_df = city_df.dropna(subset=['Date'])
                    
                    # Add city name if not present
                    if 'City' not in city_df.columns and 'city' not in city_df.columns:
                        city_name = city_file.replace('_weather_data.csv', '').title()
                        city_df['City'] = city_name
                    
                    dfs.append(city_df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Limit to recent data
                if 'Date' in combined_df.columns and not combined_df.empty:
                    try:
                        recent_date = combined_df['Date'].max() - pd.DateOffset(years=2)
                        combined_df = combined_df[combined_df['Date'] >= recent_date]
                    except:
                        # If date operations fail, just use the data as is
                        pass
                
                return combined_df
            
    except Exception as e:
        st.error(f"Error loading weather data: {str(e)}")
        return None
        
    return None

def prepare_enhanced_context(messages, weather_data, user_input):
    """
    Prepare enhanced context for AI with weather data integration
    
    Args:
        messages (list): Chat message history
        weather_data (pd.DataFrame): Weather dataset
        user_input (str): Current user query
    
    Returns:
        list: Enhanced messages with data context
    """
    try:
        # Start with system prompt
        enhanced_messages = [{"role": "system", "content": get_enhanced_system_prompt()}]
        
        # Add data context if available
        if weather_data is not None and not weather_data.empty:
            total_records = len(weather_data)
            
            # Safe date range handling
            try:
                date_range_str = f"{weather_data['Date'].min().strftime('%Y-%m-%d')} to {weather_data['Date'].max().strftime('%Y-%m-%d')}"
            except:
                date_range_str = "Date range unavailable"
            
            # Safe city list handling
            cities = weather_data['City'].unique() if 'City' in weather_data.columns else ['N/A']
            cities_str = f"{len(cities)} cities" if len(cities) > 3 else ", ".join(cities)
            
            # Safe statistics calculation
            try:
                avg_rainfall = f"{weather_data['Rainfall'].mean():.1f} mm" if 'Rainfall' in weather_data.columns else "N/A"
                avg_max_temp = f"{weather_data['Max_Temp'].mean():.1f}°C" if 'Max_Temp' in weather_data.columns else "N/A"
                avg_min_temp = f"{weather_data['Min_Temp'].mean():.1f}°C" if 'Min_Temp' in weather_data.columns else "N/A"
                avg_wind_speed = f"{weather_data['Wind_Speed'].mean():.1f} km/h" if 'Wind_Speed' in weather_data.columns else "N/A"
            except:
                avg_rainfall = avg_max_temp = avg_min_temp = avg_wind_speed = "N/A"
            
            # Recent data sample
            try:
                recent_data = weather_data.tail(5)
                recent_sample_str = recent_data.to_string(index=False, max_cols=6)
            except:
                recent_sample_str = "Recent data unavailable"
            
            data_summary = f"""
Current Weather Data Context:
- Total Records: {total_records:,}
- Date Range: {date_range_str}
- Cities: {cities_str}
- Average Rainfall: {avg_rainfall}
- Average Max Temperature: {avg_max_temp}
- Average Min Temperature: {avg_min_temp}
- Average Wind Speed: {avg_wind_speed}

Recent Data Sample:
{recent_sample_str}
"""
            
            enhanced_messages.append({
                "role": "system", 
                "content": f"CURRENT DATA CONTEXT:\n{data_summary}\n\nUser Query: {user_input}"
            })
        
        # Add conversation history (excluding system messages)
        for msg in messages:
            if msg["role"] != "system":
                enhanced_messages.append(msg)
        
        return enhanced_messages
        
    except Exception as e:
        # Fallback to basic messages if context preparation fails
        return [{"role": "system", "content": get_enhanced_system_prompt()}] + messages

def generate_smart_suggestions(messages, weather_data):
    """
    Generate smart follow-up suggestions based on conversation context
    
    Args:
        messages (list): Chat message history
        weather_data (pd.DataFrame): Weather dataset
    
    Returns:
        list: List of suggested questions
    """
    try:
        if not messages:
            return []
        
        # Get the last few messages to understand context
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        conversation_text = " ".join([msg.get("content", "") for msg in recent_messages])
        
        # Base suggestions for rainfall impact analysis
        base_suggestions = [
            "How do seasonal rainfall patterns affect crop planning in Egypt?",
            "What are the economic costs of extreme rainfall events?",
            "How does rainfall variability impact water resource management?",
            "What adaptation strategies can help with changing rainfall patterns?",
            "How do urban areas cope with heavy rainfall events?",
            "What are the environmental consequences of prolonged droughts?",
            "How does rainfall affect public health in Egyptian cities?",
            "What early warning systems exist for extreme rainfall events?"
        ]
        
        # Context-aware suggestions based on conversation
        contextual_suggestions = []
        
        if any(word in conversation_text.lower() for word in ['agriculture', 'crop', 'farm', 'yield']):
            contextual_suggestions.extend([
                "What specific crops are most vulnerable to rainfall changes?",
                "How can farmers adapt to irregular rainfall patterns?",
                "What irrigation strategies work best during dry periods?"
            ])
        
        if any(word in conversation_text.lower() for word in ['flood', 'urban', 'city', 'infrastructure']):
            contextual_suggestions.extend([
                "Which Egyptian cities have the highest flood risk?",
                "What infrastructure improvements can reduce flood damage?",
                "How do drainage systems handle extreme rainfall?"
            ])
        
        if any(word in conversation_text.lower() for word in ['economic', 'cost', 'financial', 'money']):
            contextual_suggestions.extend([
                "What are the long-term economic impacts of climate change?",
                "How much does flood damage cost Egyptian cities annually?",
                "What economic benefits come from better water management?"
            ])
        
        # Combine and prioritize suggestions
        all_suggestions = contextual_suggestions + base_suggestions
        
        # Remove duplicates and return top suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        return unique_suggestions[:6]  # Return top 6 suggestions
        
    except Exception as e:
        # Fallback suggestions if generation fails
        return [
            "How does rainfall affect agriculture in Egypt?",
            "What are the main flood risks in Egyptian cities?",
            "How can communities prepare for extreme weather events?"
        ]

def get_ai_response(messages):
    """
    Get AI response using Cerebras API
    
    Args:
        messages (list): Message history for AI context
    
    Returns:
        str: AI response or None if failed
    """
    try:
        # Initialize Cerebras client
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            return "🔑 API key not configured. Please set CEREBRAS_API_KEY in your environment variables."
        
        client = Cerebras(api_key=api_key)
        model_name = "qwen-3-coder-480b"
        
        # Get AI response
        stream = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
            max_completion_tokens=4000,
            temperature=0.7,
            top_p=0.8
        )
        
        # Collect streaming response
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        
        return full_response.strip() if full_response.strip() else "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_rainfall_visualization(user_input, weather_data):
    """
    Generate relevant rainfall visualizations based on user query
    
    Args:
        user_input (str): User's query
        weather_data (pd.DataFrame): Weather dataset
    
    Returns:
        plotly.graph_objects.Figure or None: Generated chart
    """
    try:
        if weather_data is None or weather_data.empty:
            return None
        
        query_lower = user_input.lower()
        
        # Rainfall trend analysis
        if any(word in query_lower for word in ['trend', 'pattern', 'time', 'change', 'rainfall']):
            if 'Rainfall' in weather_data.columns and 'Date' in weather_data.columns:
                fig = go.Figure()
                
                # Monthly aggregation
                weather_data['Month'] = weather_data['Date'].dt.to_period('M')
                monthly_rainfall = weather_data.groupby('Month')['Rainfall'].mean().reset_index()
                monthly_rainfall['Month'] = monthly_rainfall['Month'].astype(str)
                
                fig.add_trace(go.Scatter(
                    x=monthly_rainfall['Month'],
                    y=monthly_rainfall['Rainfall'],
                    mode='lines+markers',
                    name='Monthly Average Rainfall',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title='Monthly Rainfall Trends',
                    xaxis_title='Month',
                    yaxis_title='Rainfall (mm)',
                    height=400
                )
                
                return fig
        
        # City comparison
        if any(word in query_lower for word in ['city', 'cities', 'compare', 'comparison']) and 'City' in weather_data.columns:
            city_rainfall = weather_data.groupby('City')['Rainfall'].mean().reset_index()
            city_rainfall = city_rainfall.sort_values('Rainfall', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=city_rainfall['City'],
                    y=city_rainfall['Rainfall'],
                    marker_color='lightblue',
                    text=city_rainfall['Rainfall'].round(1),
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Average Rainfall by City',
                xaxis_title='City',
                yaxis_title='Average Rainfall (mm)',
                height=400
            )
            
            return fig
        
        # Seasonal patterns
        if any(word in query_lower for word in ['season', 'monthly', 'month', 'seasonal']):
            if 'Date' in weather_data.columns and 'Rainfall' in weather_data.columns:
                weather_data['Month'] = weather_data['Date'].dt.month
                seasonal_rainfall = weather_data.groupby('Month')['Rainfall'].mean().reset_index()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                seasonal_rainfall['Month_Name'] = seasonal_rainfall['Month'].map(
                    {i+1: month_names[i] for i in range(12)}
                )
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=seasonal_rainfall['Month_Name'],
                        y=seasonal_rainfall['Rainfall'],
                        marker_color='green',
                        text=seasonal_rainfall['Rainfall'].round(1),
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title='Seasonal Rainfall Patterns',
                    xaxis_title='Month',
                    yaxis_title='Average Rainfall (mm)',
                    height=400
                )
                
                return fig
        
        return None
        
    except Exception as e:
        return None

def preprocess_data(df):
    """
    Preprocess the input DataFrame for machine learning
    
    Args:
        df (pd.DataFrame): Input weather DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    df = df.sort_values('Date').reset_index(drop=True)
    df['is_rain'] = (df['Rainfall_mm'] > 0).astype(int)

    # Time features
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday

    # Lags and rolling
    for lag in [1, 2, 3, 4, 5]:
        df[f'lag{lag}'] = df['Rainfall_mm'].shift(lag)
    df['roll3'] = df['Rainfall_mm'].rolling(3).mean()
    df['roll5'] = df['Rainfall_mm'].rolling(5).mean()
    df['roll7'] = df['Rainfall_mm'].rolling(7).mean()
    df['diff1'] = df['Rainfall_mm'].diff(1)

    # Additional features
    df['Temp_Diff'] = df['Temp_Max_C'] - df['Temp_Min_C']
    df['Wind_Rain_Lag1'] = df['WindSpeed_Max_kmh'] * df['lag1']

    # Drop NAs
    df = df.dropna().reset_index(drop=True)
    return df

def time_split(df, split_ratio=0.8):
    """
    Split data into train and test sets based on time
    
    Args:
        df (pd.DataFrame): Input DataFrame
        split_ratio (float): Proportion of data to use for training
    
    Returns:
        tuple: Train and test DataFrames
    """
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test
 
def train_models(train, features):
    """
    Train classification and regression models
    
    Args:
        train (pd.DataFrame): Training DataFrame
        features (list): List of feature column names
    
    Returns:
        tuple: Trained classifier and regressor models
    """
    clf = LGBMClassifier(random_state=42)
    clf.fit(train[features], train['is_rain'])

    reg = LGBMRegressor(random_state=42)
    rain_data = train[train['is_rain'] == 1]
    reg.fit(rain_data[features], rain_data['Rainfall_mm'])

    return clf, reg

def forecast(clf, reg, test, features, threshold=0.5):
    """
    Generate rainfall forecasts
    
    Args:
        clf (LGBMClassifier): Trained classifier
        reg (LGBMRegressor): Trained regressor
        test (pd.DataFrame): Test DataFrame
        features (list): List of feature column names
        threshold (float): Probability threshold for rain prediction
    
    Returns:
        tuple: Forecast values, rain predictions, and probabilities
    """
    probs = clf.predict_proba(test[features])[:, 1]
    rain_pred = (probs >= threshold).astype(int)
    rain_amt = reg.predict(test[features])
    final = np.where(rain_pred == 1, rain_amt, 0.0)
    return final, rain_pred, probs

def render_forecasting_chatbot_page():
    """
    Render the Rainfall Forecasting page
    """
    # Add a tab for AI Chatbot
    tab1, tab2 = st.tabs(["🌧️ Rainfall Forecast", "🤖 AI Insights"])
    
    with tab1:
        st.markdown('<h1 class="main-header">🔮 Rainfall Forecasting</h1>', unsafe_allow_html=True)
        
        # Discover available data files
        data_files = [f for f in os.listdir('src/data') if f.endswith('_weather_data.csv')]
        
        # City selection
        st.sidebar.markdown("## 🌆 Forecasting Setup")
        selected_city = st.sidebar.selectbox(
            "Select City for Forecasting", 
            [f.replace('_weather_data.csv', '').title() for f in data_files]
        )
        
        # Load selected city's data
        city_file = f'{selected_city.lower()}_weather_data.csv'
        try:
            df = pd.read_csv(f'src/data/{city_file}')
            df['Date'] = pd.to_datetime(df['Date'])
        except FileNotFoundError:
            st.error(f"No data found for {selected_city}. Please collect data first.")
            return
        
        # Forecasting parameters
        st.sidebar.markdown("## ⚙️ Model Parameters")
        split_ratio = st.sidebar.slider(
            "Train-Test Split Ratio", 
            min_value=0.6, 
            max_value=0.9, 
            value=0.8, 
            step=0.05
        )
        rain_threshold = st.sidebar.slider(
            "Rain Probability Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05
        )
        
        # Preprocessing
        features = [
            'Temp_Max_C', 'Temp_Min_C', 'WindSpeed_Max_kmh', 
            'dayofyear', 'month', 'weekday'
        ]
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Split data
        train, test = time_split(processed_df, split_ratio)
        
        # Extend features
        features.extend([
            'Temp_Diff', 'lag1', 'lag2', 'lag3', 
            'roll3', 'roll5', 'Wind_Rain_Lag1'
        ])
        
        # Train models
        clf, reg = train_models(train, features)
            
        # Forecast
        forecast_vals, rain_pred, rain_probs = forecast(
            clf, reg, test, features, threshold=rain_threshold
        )
        
        # Prepare results
        results = test[['Date', 'Rainfall_mm']].copy()
        results['Predicted_Rainfall'] = forecast_vals
        results['Rain_Probability'] = rain_probs
        results['Rain_Prediction'] = rain_pred
        
        # Add is_rain column
        results['is_rain'] = (results['Rainfall_mm'] > 0).astype(int)
        
        # Metrics
        st.markdown("## 📊 Forecasting Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            rmse = np.sqrt(np.mean((results['Rainfall_mm'] - results['Predicted_Rainfall'])**2))
            st.metric("RMSE", f"{rmse:.2f} mm")
        
        with col2:
            accuracy = np.mean(results['Rain_Prediction'] == results['is_rain'])
            st.metric("Rain Prediction Accuracy", f"{accuracy:.2%}")
        
        # Global Plotly Template
        plotly_template = 'plotly_white'
        
        # Color Palette
        colors = {
            'primary': '#3498db',    # Bright Blue
            'secondary': '#2ecc71',  # Emerald Green
            'accent': '#e74c3c',     # Vibrant Red
            'background': '#f4f6f7', # Light Gray Background
            'text': '#2c3e50'        # Dark Blue-Gray Text
        }
        
        # Visualization Styling Function
        def style_plotly_figure(fig, title, x_title, y_title):
            """
            Apply consistent styling to Plotly figures
            """
            fig.update_layout(
                template=plotly_template,
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center', 
                    'yanchor': 'top',
                    'font': dict(size=20, color=colors['text'])
                },
                xaxis_title=x_title,
                yaxis_title=y_title,
                font=dict(family="Arial, sans-serif", size=12, color=colors['text']),
                plot_bgcolor=colors['background'],
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=50),
                height=400,
                hovermode='x unified'
            )
            
            # Styling axis
            fig.update_xaxes(
                showline=True, 
                linewidth=2, 
                linecolor='lightgray', 
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                showline=True, 
                linewidth=2, 
                linecolor='lightgray', 
                gridcolor='lightgray'
            )
            
            return fig
        
        # Visualizations
        st.markdown("## 📈 Forecast Visualization")
        
        # Actual vs Predicted Rainfall 
        st.markdown("### 🌧️ Rainfall Forecast")
        fig_rainfall = go.Figure()
        
        # Actual Rainfall
        fig_rainfall.add_trace(go.Scatter(
            x=results['Date'], 
            y=results['Rainfall_mm'], 
            mode='lines', 
            name='Actual Rainfall',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Rainfall: %{y:.2f} mm<extra></extra>'
        ))
        
        # Predicted Rainfall
        fig_rainfall.add_trace(go.Scatter(
            x=results['Date'], 
            y=results['Predicted_Rainfall'], 
            mode='lines', 
            name='Predicted Rainfall',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>Predicted Rainfall: %{y:.2f} mm<extra></extra>'
        ))
        
        # Confidence Interval (using standard deviation)
        std_dev = results['Predicted_Rainfall'].std()
        fig_rainfall.add_trace(go.Scatter(
            x=results['Date'].tolist() + results['Date'].tolist()[::-1],
            y=(results['Predicted_Rainfall'] + std_dev).tolist() + 
               (results['Predicted_Rainfall'] - std_dev).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig_rainfall.update_layout(
            title=f'Rainfall Forecast for {selected_city}',
            xaxis_title='Date',
            yaxis_title='Rainfall (mm)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_rainfall, use_container_width=True)
        
        # Advanced Precipitation Patterns Analysis
        st.markdown("### 🌈 Precipitation Patterns")
        
        # Prepare data for advanced visualization
        results['Year'] = results['Date'].dt.year
        results['Month'] = results['Date'].dt.month
        results['Day'] = results['Date'].dt.day
        
        # Create a multi-layered precipitation visualization
        fig_precip = go.Figure()
        
        # Cumulative Rainfall Line
        cumulative_rainfall = results['Rainfall_mm'].cumsum()
        fig_precip.add_trace(go.Scatter(
            x=results['Date'],
            y=cumulative_rainfall,
            mode='lines',
            name='Cumulative Rainfall',
            line=dict(color='blue', width=3),
            hovertemplate='Date: %{x}<br>Cumulative Rainfall: %{y:.2f} mm<extra></extra>'
        ))
        
        # Daily Rainfall Bar Chart
        fig_precip.add_trace(go.Bar(
            x=results['Date'],
            y=results['Rainfall_mm'],
            name='Daily Rainfall',
            marker_color='skyblue',
            opacity=0.5,
            hovertemplate='Date: %{x}<br>Daily Rainfall: %{y:.2f} mm<extra></extra>'
        ))
        
        # Precipitation Probability Overlay
        fig_precip.add_trace(go.Scatter(
            x=results['Date'],
            y=results['Rain_Probability'] * results['Rainfall_mm'].max(),
            mode='lines',
            name='Rain Probability',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>Rain Probability: %{y:.2%}<extra></extra>'
        ))
        
        # Extreme Rainfall Events Markers
        extreme_events = results[results['Rainfall_mm'] > results['Rainfall_mm'].quantile(0.95)]
        fig_precip.add_trace(go.Scatter(
            x=extreme_events['Date'],
            y=extreme_events['Rainfall_mm'],
            mode='markers',
            name='Extreme Rainfall Events',
            marker=dict(
                color='red', 
                size=extreme_events['Rainfall_mm'] * 2,
                sizemode='area',
                sizeref=2.*max(extreme_events['Rainfall_mm'])/(40.**2),
                sizemin=4,
                line=dict(width=1, color='darkred')
            ),
            hovertemplate='Date: %{x}<br>Extreme Rainfall: %{y:.2f} mm<extra></extra>'
        ))
        
        # Layout
        fig_precip.update_layout(
            title='Comprehensive Precipitation Analysis',
            height=600,
            xaxis_title='Date',
            yaxis_title='Rainfall (mm)',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        st.plotly_chart(fig_precip, use_container_width=True)
        
        # Precipitation Seasonality and Variability
        st.markdown("### 📊 Precipitation Seasonality")
        
        # Monthly Aggregation
        monthly_rainfall = results.groupby('Month')['Rainfall_mm'].agg(['mean', 'max', 'count']).reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_rainfall['Month_Name'] = monthly_rainfall['Month'].map(
            {i+1: month_names[i] for i in range(12)}
        )
        
        # Seasonal Rainfall Bar Chart
        fig_seasonal = go.Figure(data=[
            go.Bar(
                x=monthly_rainfall['Month_Name'],
                y=monthly_rainfall['mean'],
                name='Average Rainfall',
                error_y=dict(
                    type='data',
                    array=monthly_rainfall['max'] - monthly_rainfall['mean'],
                    visible=True,
                    color='red'
                ),
                hovertemplate='Month: %{x}<br>Avg Rainfall: %{y:.2f} mm<br>Max Rainfall: %{error_y.array:.2f} mm<extra></extra>'
            )
        ])
        
        fig_seasonal.update_layout(
            title='Monthly Rainfall Distribution',
            xaxis_title='Month',
            yaxis_title='Average Rainfall (mm)',
            height=500
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Precipitation Insights
        st.markdown("### 🌦️ Precipitation Insights")
        
        # Create columns for insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total Precipitation
            total_rainfall = results['Rainfall_mm'].sum()
            st.metric("Total Rainfall", f"{total_rainfall:.2f} mm")
        
        with col2:
            # Rainy Days
            rainy_days = (results['Rainfall_mm'] > 0).sum()
            total_days = len(results)
            st.metric("Rainy Days", f"{rainy_days} / {total_days}")
        
        with col3:
            # Maximum Daily Rainfall
            max_daily_rainfall = results['Rainfall_mm'].max()
            st.metric("Max Daily Rainfall", f"{max_daily_rainfall:.2f} mm")
        
        with col4:
            # Rainfall Variability
            rainfall_std = results['Rainfall_mm'].std()
            st.metric("Rainfall Variability", f"{rainfall_std:.2f} mm")
        
        # Feature Importance 
        st.markdown("### 🔍 Feature Importance")
        # For Classification Model
        clf_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
        
        fig_clf_imp = go.Figure(go.Bar(
            x=clf_importances.values,
            y=clf_importances.index,
            orientation='h',
            marker_color='green',
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig_clf_imp.update_layout(
            title='Classification Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500
        )
        st.plotly_chart(fig_clf_imp, use_container_width=True)
        
        # Detailed Performance Metrics Visualization
        st.markdown("## 🎯 Detailed Performance Metrics")
        
        # Confusion Matrix with Detailed Annotations
        cm = confusion_matrix(results['is_rain'], results['Rain_Prediction'])
        
        # Calculate performance metrics
        tn, fp, fn, tp = cm.ravel()
        
        # Performance Metrics Calculation
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Confusion Matrix Visualization
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['No Rain', 'Rain'],
            y=['No Rain', 'Rain'],
            hoverongaps=False,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size':14},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix: Rain Prediction Performance',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Performance Metrics Visualization
        fig_metrics = go.Figure()
        
        # Bar chart of performance metrics
        metrics_data = [
            {'Metric': 'Accuracy', 'Value': accuracy},
            {'Metric': 'Precision', 'Value': precision},
            {'Metric': 'Recall', 'Value': recall},
            {'Metric': 'F1 Score', 'Value': f1_score}
        ]
        
        fig_metrics.add_trace(go.Bar(
            x=[metric['Metric'] for metric in metrics_data],
            y=[metric['Value'] for metric in metrics_data],
            text=[f'{metric["Value"]:.2%}' for metric in metrics_data],
            textposition='outside',
            marker_color=['blue', 'green', 'red', 'purple'],
            hovertemplate='%{x}<br>Value: %{y:.2%}<extra></extra>'
        ))
        
        fig_metrics.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            height=500
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Detailed Metrics Section
        st.markdown("### 📊 Detailed Prediction Analysis")
        
        # Create columns for insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # True Negatives (Correctly Predicted No Rain)
            st.metric("True Negatives", f"{tn}")
        
        with col2:
            # False Positives (Incorrectly Predicted Rain)
            st.metric("False Positives", f"{fp}")
        
        with col3:
            # False Negatives (Missed Rain Events)
            st.metric("False Negatives", f"{fn}")
        
        with col4:
            # True Positives (Correctly Predicted Rain)
            st.metric("True Positives", f"{tp}")
        
        # Additional Insights
        st.markdown("### 🔍 Prediction Probability Analysis")
        
        # Probability Distribution
        fig_prob_dist = go.Figure()
        
        # Histogram of Rain Probabilities
        fig_prob_dist.add_trace(go.Histogram(
            x=results['Rain_Probability'],
            nbinsx=50,
            name='Probability Distribution',
            marker_color='skyblue',
            opacity=0.7,
            hovertemplate='Probability Range: %{x:.2f}<br>Frequency: %{y}<extra></extra>'
        ))
        
        # Actual Rain Events
        rain_events = results[results['Rainfall_mm'] > 0]
        no_rain_events = results[results['Rainfall_mm'] == 0]
        
        # Vertical lines for different thresholds
        thresholds = [0.3, 0.5, 0.7]
        colors = ['green', 'orange', 'red']
        
        for threshold, color in zip(thresholds, colors):
            # Count of events at each threshold
            events_above_threshold = results[results['Rain_Probability'] >= threshold]
            actual_rain_above_threshold = events_above_threshold[events_above_threshold['Rainfall_mm'] > 0]
            
            # Add vertical line
            fig_prob_dist.add_shape(
                type='line',
                x0=threshold,
                x1=threshold,
                y0=0,
                y1=len(results),
                line=dict(color=color, dash='dot')
            )
            
            # Annotation for threshold details
            fig_prob_dist.add_annotation(
                x=threshold,
                y=len(results),
                text=f'Threshold {threshold:.1f}<br>Total Events: {len(events_above_threshold)}<br>Actual Rain: {len(actual_rain_above_threshold)}',
                showarrow=True,
                arrowhead=1
            )
        
        fig_prob_dist.update_layout(
            title='Rain Probability Distribution with Threshold Analysis',
            xaxis_title='Rain Probability',
            yaxis_title='Frequency',
            height=600
        )
        
        st.plotly_chart(fig_prob_dist, use_container_width=True)
        
        # Time Series Decomposition
        st.markdown("### 🕰️ Time Series Decomposition")
        
        # Import statsmodels for time series decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare time series
        ts_rainfall = results.set_index('Date')['Rainfall_mm']
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(ts_rainfall, period=30)  # Assuming monthly seasonality
            
            # Create decomposition plot
            fig_decomp = go.Figure()
            
            # Original Series
            fig_decomp.add_trace(go.Scatter(
                x=ts_rainfall.index,
                y=ts_rainfall.values,
                mode='lines',
                name='Original',
                line=dict(color='blue')
            ))
            
            # Trend
            fig_decomp.add_trace(go.Scatter(
                x=ts_rainfall.index,
                y=decomposition.trend,
                mode='lines',
                name='Trend',
                line=dict(color='green', dash='dot')
            ))
            
            # Seasonal Component
            fig_decomp.add_trace(go.Scatter(
                x=ts_rainfall.index,
                y=decomposition.seasonal,
                mode='lines',
                name='Seasonal',
                line=dict(color='red', dash='dash')
            ))
            
            # Residual
            fig_decomp.add_trace(go.Scatter(
                x=ts_rainfall.index,
                y=decomposition.resid,
                mode='lines',
                name='Residual',
                line=dict(color='purple', dash='dot')
            ))
            
            fig_decomp.update_layout(
                title='Time Series Decomposition of Rainfall',
                height=600,
                xaxis_title='Date',
                yaxis_title='Rainfall (mm)'
            )
            
            st.plotly_chart(fig_decomp, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not perform time series decomposition: {e}") 

    with tab2:
        # Render AI Chatbot
        render_ai_chatbot() 

# API Key Security Warning
def check_api_key():
    """
    Check and warn about API key security
    """
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        warnings.warn(
            "🚨 SECURITY WARNING 🚨\n"
            "Cerebras API key not found in environment variables.\n"
            "Please create a .env file in your project root with:\n"
            "CEREBRAS_API_KEY=your_api_key_here\n"
            "IMPORTANT: Never share your API key publicly or commit it to version control!",
            UserWarning
        )
    return api_key

# Call this function early in the script
check_api_key() 

# Modify the existing import and add explicit loading
from dotenv import load_dotenv, dotenv_values
import os
import warnings

# Load environment variables explicitly
def load_environment_variables():
    """
    Explicitly load environment variables from .env file
    """
    # First, try to load from .env file
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not api_key:
        # If not loaded, try reading directly from .env
        env_path = os.path.join(os.getcwd(), '.env')
        env_vars = dotenv_values(env_path)
        api_key = env_vars.get("CEREBRAS_API_KEY")
        
        if api_key:
            # Manually set the environment variable
            os.environ["CEREBRAS_API_KEY"] = api_key
    
    # Warn if API key is still not found
    if not api_key:
        warnings.warn(
            "🚨 SECURITY WARNING 🚨\n"
            "Cerebras API key not found. Please check your .env file.\n"
            "Create a .env file in your project root with:\n"
            "CEREBRAS_API_KEY=your_api_key_here",
            UserWarning
        )
    
    return api_key

# Call the function to load environment variables
load_environment_variables() 