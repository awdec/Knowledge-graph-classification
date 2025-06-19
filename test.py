import openai

client = openai.OpenAI(
    api_key="sk-93qverYBR6IKSZnvFQ9Rn7lN1Enm3P8BM2KBZ8aMwaHqywnP",
    base_url='https://api.openai-proxy.org/v1',  # 如果你用的是 CloseAI
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好，请介绍一下杭州"}
    ]
)

print(response.choices[0].message.content)
