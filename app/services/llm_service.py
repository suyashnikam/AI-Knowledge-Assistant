from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_response(messages):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.1",
        messages=messages
    )
    return response.choices[0].message.content