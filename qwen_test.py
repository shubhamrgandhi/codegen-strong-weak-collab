# from openai import OpenAI
# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8080/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     messages=[
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": "Tell me something about large language models."},
#     ],
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=512,
#     extra_body={
#         "repetition_penalty": 1.05,
#     },
# )
# print("Chat response:", chat_response)


# ===============================

from openai import OpenAI
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"],
)

completion = client.chat.completions.create(
  model="deepseek/deepseek-r1-distill-llama-70b:free",
  temperature=0.0,
  n=1,
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)