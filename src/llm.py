from openai import OpenAI

def query(prompt, model):
    key = 'xx'
    base_url="xx"
    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,     
        messages=[{"role": "user",
                   "content": prompt},
                  ]
    )
    answer = resp.choices[0].message.content
    return answer.strip()


