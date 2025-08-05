def generate_weather_response(question, data=None):
    """
    Generate intelligent responses to weather-related questions
    
    Args:
        question (str): User's input question
        data (pd.DataFrame, optional): Weather dataset for context
    
    Returns:
        str: Generated response
    """
    # Convert question to lowercase for pattern matching
    q_lower = question.lower()

    # Sample responses based on keywords
    if "wettest" in q_lower and "year" in q_lower:
        return """Based on the historical data from 2000-2024, I can help you analyze the wettest years.

**Key Insights:**
- Wettest years vary by city due to different climate zones
- Northern cities typically receive more rainfall
- Seasonal patterns show most rainfall during winter months

*Note: Actual analysis requires loading the complete dataset.*"""

    elif "compare" in q_lower and ("rainfall" in q_lower or "temperature" in q_lower):
        return """I can help you compare weather patterns between cities!

**Climate Comparison Example:**
🌊 **Coastal Cities** (e.g., Alexandria):
- Higher annual rainfall
- Milder temperature variations
- More humid conditions

🏜️ **Desert Cities** (e.g., Aswan):
- Minimal rainfall
- Higher temperature extremes
- Very dry conditions

*Detailed comparisons require loading the dataset.*"""

    else:
        return """Hello! I'm your Weather Intelligence Assistant.

🤖 **I can help you with:**
- Historical weather statistics
- City-by-city climate comparisons
- Seasonal pattern analysis
- Drought and extreme weather insights

Try asking about specific cities, time periods, or weather phenomena!

*Note: Full responses require loading the weather dataset.*""" 