import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_chat_completion(chat_history: list) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        chat_history,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 2048
        }
    )
    return response.text