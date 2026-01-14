import requests
import json
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

TOOLS_DESCRIPTION = """
You are a restaurant reservation assistant. You have access to these functions:

1. make_reservation
   - Purpose: Create a new reservation
   - Parameters:
     * name (string, required): Customer's full name
     * date (string, required): Reservation date in YYYY-MM-DD format
     * time (string, required): Reservation time (e.g., "19:00" or "7:00 PM")
     * party_size (integer, required): Number of people (1-20)

2. list_reservations
   - Purpose: Show all current reservations
   - Parameters: None

3. cancel_reservation
   - Purpose: Cancel an existing reservation
   - Parameters:
     * reservation_id (integer, required): The ID number of the reservation to cancel

RESPONSE FORMAT:
You must respond with ONLY a valid JSON object in this exact format:

{
  "function_name": "the_function_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  },
  "reasoning": "Brief explanation of why you chose this function"
}

If no function is needed (e.g., user just says hello), use:
{
  "function_name": "none",
  "response": "Your friendly response here",
  "reasoning": "Why no function was needed"
}

IMPORTANT RULES:
- Return ONLY valid JSON, no extra text
- Use "none" as function_name if no function should be called
- Extract all required parameters accurately
- Convert dates to YYYY-MM-DD format
- Convert times to 24-hour format if possible
"""

def ask_llm_which_function_to_call(user_message):
    if OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("OPENROUTER_API_KEY not set")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        #User message and tool description are passed as input to LLM
        "messages": [
            {"role": "system", "content": TOOLS_DESCRIPTION},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        #Send request to OpenRouter API
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        #Parse LLM response as JSON
        response_data = response.json()
        llm_response_text = response_data['choices'][0]['message']['content'].strip()
        
        #Clean up response to extract JSON
        if llm_response_text.startswith("```json"):
            llm_response_text = llm_response_text.replace("```json", "").replace("```", "").strip()
        elif llm_response_text.startswith("```"):
            llm_response_text = llm_response_text.replace("```", "").strip()
        
        #Convert response text to dictionary
        llm_decision = json.loads(llm_response_text)
        
        #Validate presence of function_name
        if 'function_name' not in llm_decision:
            raise ValueError("LLM response missing 'function_name' field")
        
        return llm_decision #Back to main.py
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to call OpenRouter API: {str(e)}")
    
    except json.JSONDecodeError as e:
        raise Exception(f"LLM returned invalid JSON: {str(e)}")
    
    except Exception as e:
        raise