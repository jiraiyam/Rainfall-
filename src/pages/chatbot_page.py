import streamlit as st
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import os
import time
import re

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(page_title="Rainfall Insights Chatbot", page_icon="🌧️")

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.title("🌧️ Rainfall Insights Chatbot")

# Initialize variables
api_key = None
client = None
model_name = "qwen-3-coder-480b"

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

try:
    # Get API key
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        st.error("CEREBRAS_API_KEY not found in environment variables.")
        st.stop()

    # Initialize client
    client = Cerebras(api_key=api_key)

    # Verify model availability
    try:
        model_list = client.models.list()
        available_models = [model.name for model in model_list.data]
        if model_name not in available_models:
            st.warning(f"⚠️ Model '{model_name}' not available. Available models: {', '.join(available_models)}")
    except Exception as e:
        st.warning(f"⚠️ Could not verify model availability: {str(e)}")

    st.caption(f"Powered by Cerebras Cloud - {model_name}")

except Exception as e:
    st.error(f"🚨 Failed to initialize Cerebras client: {str(e)}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": get_system_prompt()}
    ]

# Display chat history - excluding system message
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about rainfall, weather, or climate..."):
    # Check if query is weather-related
    if is_weather_related_query(prompt):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            error_occurred = False

            try:
                # Create chat completion with streaming
                stream = client.chat.completions.create(
                    messages=st.session_state.messages,
                    model=model_name,
                    stream=True,
                    max_completion_tokens=4000,
                    temperature=0.7,
                    top_p=0.8
                )

                # Stream the response
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)  # Smooth out streaming

                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
                error_occurred = True
                full_response = f"Error: {str(e)}"

            # Add assistant response to history if successful
            if not error_occurred:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Non-weather related query
        st.warning("🌧️ Sorry, I can only discuss topics related to rainfall, weather, and climate. "
                   "Please ask a question about precipitation, meteorology, or climate patterns.")

# Sidebar controls
with st.sidebar:
    st.header("Rainfall Insights Configuration")

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": get_system_prompt()}
        ]
        st.rerun()

    st.divider()
    st.subheader("Current Session")
    st.write(f"**Messages:** {len(st.session_state.messages) - 1}")  # Exclude system message
    st.write(f"**Model:** {model_name}")

    st.divider()
    st.subheader("API Parameters")
    st.write(f"**Max tokens:** 4000")
    st.write(f"**Temperature:** 0.7")
    st.write(f"**Top-p:** 0.8")

    st.divider()
    st.caption("Specialized in rainfall and weather insights")
    st.caption("Powered by advanced AI technology")