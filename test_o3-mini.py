import openai
import os

# Retrieve the API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()
response = client.chat.completions.create(
    model='o3-mini-2025-01-31',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain the concept of recursion in programming.'}
    ],
    reasoning_effort='high'
)

print('-------------------- RESPONSE -------------------- \n\n')
print(response)

chain_of_thought = response.choices[0].message['content']

print('\n\n-------------------- CHAIN OF THOUGHT -------------------- \n\n')
print(chain_of_thought)
