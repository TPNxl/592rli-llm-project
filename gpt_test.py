from openai import OpenAI

with open("./tokens/openai_token.txt", 'r') as f:
    token = f.read().strip()

client = OpenAI(api_key=token)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is Machine Learning?",
        }
    ],
    model="gpt-4o-mini",
)

print(chat_completion.choices[0].message.content)