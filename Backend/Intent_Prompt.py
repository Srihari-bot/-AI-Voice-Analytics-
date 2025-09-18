"""
Intent Recognition System Message for Business Inquiries
This module contains the system prompt for intent recognition using IBM WatsonX
"""

INTENT_SYSTEM_MESSAGE = """
    You are an expert Tally support agent. Your single task is to analyze the user's problem from the Transcript and, using the provided Context from the official Tally FAQ, generate a single, valid JSON object with one key:

identified_intent: A short, crisp, and actionable summary of the user's core problem.

Rules:

You MUST base your analysis ONLY on the provided Context and Transcript.

The identified_intent should be a summary statement, not a question.

Your response MUST BE ONLY a single, valid JSON object. Do not add any other words, explanation, or formatting.

Context from Tally FAQ:
{context}

Example of a perfect output:

json
{
  "identified_intent": "GSTR1 report mismatch with sales register"
}
User Problem (Transcript):
"{transcribed_text}"

Your JSON Output:
"""

def get_intent_system_message():
    """
    Returns the intent recognition system message.
    
    Returns:
        str: The system message for intent recognition
    """
    return INTENT_SYSTEM_MESSAGE

def get_intent_user_message(text_content):
    """
    Creates the user message for intent recognition.
    
    Args:
        text_content (str): The transcribed text to analyze
        
    Returns:
        str: The formatted user message
    """
    return f"Identify the intent in this transcribed text: {text_content}"

# ADD THIS MISSING FUNCTION
def extract_intent_category(intent: str) -> str:
    """
    Extract category from intent for specialized prompts
    
    Args:
        intent (str): The identified intent text
        
    Returns:
        str: The category classification
    """
    intent_lower = intent.lower()
    
    if any(word in intent_lower for word in ['gst', 'tax', 'return', 'filing', 'gstr']):
        return 'gst'
    elif any(word in intent_lower for word in ['license', 'activation', 'key', 'registration', 'site id', 'account id']):
        return 'licensing'
    elif any(word in intent_lower for word in ['error', 'crash', 'problem', 'issue', 'bug', 'not working']):
        return 'technical'
    elif any(word in intent_lower for word in ['how to', 'navigate', 'find', 'access', 'where is']):
        return 'navigation'
    elif any(word in intent_lower for word in ['report', 'generate', 'print', 'export']):
        return 'reporting'
    elif any(word in intent_lower for word in ['data', 'backup', 'restore', 'import', 'export']):
        return 'data_management'
    else:
        return 'general'
