import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("/local/yzheng/pnair/.env")

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment variables.")

client = openai.OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    "Decompose the following question into a set of sub-questions to find the answer to this main question. Also determine an expected answer type for the main question:\n"
                    "Main question: Where was the designer of the Lap Engine educated?"
                ),
            }
        ],
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error communicating with OpenAI API: {e}")