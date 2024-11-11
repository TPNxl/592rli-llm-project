import asyncio
from openai import AsyncOpenAI


with open("./tokens/openai_token.txt", 'r') as f:
    token = f.read().strip()


client = AsyncOpenAI(api_key=token)

# Define a list of different prompts you want to send
prompts = [
    "What's the capital of France?",
    "Explain the concept of quantum entanglement.",
    "Summarize the plot of Romeo and Juliet."
]

# Asynchronous function to handle each request
async def get_response(prompt):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Main function to gather all responses asynchronously
async def batch_requests(prompts):
    tasks = [get_response(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Run the asynchronous function to get responses for all prompts
responses = asyncio.run(batch_requests(prompts))

# Print each response
for i, response in enumerate(responses):
    print(f"Response {i + 1}:\n{response}\n")