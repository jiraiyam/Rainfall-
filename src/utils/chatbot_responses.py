import random

class WeatherChatbot:
    def __init__(self, data=None):
        """
        Initialize the chatbot with optional weather data
        
        Args:
            data (pd.DataFrame, optional): Weather dataset for context
        """
        self.data = data
    
    def generate_response(self, question):
        """
        Generate intelligent responses to weather-related questions
        
        Args:
            question (str): User's input question
        
        Returns:
            str: Generated response
        """
        # Convert question to lowercase for pattern matching
        q_lower = question.lower()
        
        # Predefined response categories
        responses = {
            "wettest_year": [
                "Based on historical data, the wettest years can vary by city.",
                "Northern cities like Alexandria typically experience more rainfall.",
                "Winter months (December-February) usually have the highest precipitation."
            ],
            "city_comparison": [
                "Each Egyptian city has a unique climate profile.",
                "Coastal cities have different rainfall patterns compared to inland regions.",
                "Factors like proximity to the Mediterranean and elevation impact weather."
            ],
            "temperature_trend": [
                "Temperature trends reveal fascinating climate dynamics.",
                "Urban heat island effects can influence city temperatures.",
                "Long-term data helps us understand climate change impacts."
            ],
            "drought": [
                "Drought analysis is crucial for understanding climate resilience.",
                "Southern cities like Aswan experience more prolonged dry periods.",
                "Drought is typically defined by extended periods of minimal rainfall."
            ],
            "seasonal_patterns": [
                "Egyptian weather follows distinct seasonal rhythms.",
                "Each season brings unique meteorological characteristics.",
                "Understanding seasonal patterns helps in agriculture and planning."
            ],
            "default": [
                "Interesting question! Weather data can reveal fascinating insights.",
                "Climate analysis helps us understand complex environmental systems.",
                "Every weather pattern tells a story about our changing planet."
            ]
        }
        
        # Keyword matching for response selection
        if any(keyword in q_lower for keyword in ["wettest", "rainfall", "precipitation"]):
            category = "wettest_year"
        elif "compare" in q_lower:
            category = "city_comparison"
        elif "temperature" in q_lower or "temp" in q_lower:
            category = "temperature_trend"
        elif "drought" in q_lower:
            category = "drought"
        elif "seasonal" in q_lower or "pattern" in q_lower:
            category = "seasonal_patterns"
        else:
            category = "default"
        
        # Select a random response from the chosen category
        base_response = random.choice(responses[category])
        
        # If data is available, enhance response with specific insights
        if self.data is not None:
            # Add data-driven insights here
            pass
        
        return base_response
    
    def suggest_questions(self):
        """
        Generate a list of suggested questions for the chatbot
        
        Returns:
            list: Suggested weather-related questions
        """
        return [
            "What was the wettest year in Cairo?",
            "Compare rainfall between Alexandria and Aswan",
            "Show me temperature trends over the last decade",
            "Which city has the most variable weather?",
            "When is the drought season in Egypt?",
            "What are the seasonal patterns for wind speed?",
            "How has climate changed since 2000?",
            "Which months have the highest rainfall?"
        ] 