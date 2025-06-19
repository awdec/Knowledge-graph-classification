import openai

client = openai.OpenAI(
    api_key="your api key",
    base_url='https://api.openai-proxy.org/v1',  # 如果你用的是 CloseAI
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好，请介绍一下杭州"}
    ]
)

print(response.choices[0].message.content)
