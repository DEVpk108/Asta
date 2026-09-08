import os
from openai import OpenAI

# Initialize the client with CodeCraft's base URL and your API key
client = OpenAI(
    api_key="sk-30fa1f6bc03b474283cf391ad2594bbd",  # Replace with your actual CodeCraft API key
    base_url="https://api.deepseek.com/",
)

# Call one of the models returned in your earlier list
response = client.chat.completions.create(
    model="deepseek-v4-flash",  # Or "gemini-3.7-flash", "gpt-5.6-luna", etc.
    messages=[
        {"role": "user", "content": "Hello! Write a short 1-sentence greeting."}
    ],
)

# Print the response content
print(response.choices[0].message.content)