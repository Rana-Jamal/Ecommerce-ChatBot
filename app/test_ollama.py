from ollama_config import chat_completion

answer = chat_completion(
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    temperature=0.2
)

print(answer)