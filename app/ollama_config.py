import os
from dotenv import load_dotenv
from ollama import Client


load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

if not OLLAMA_API_KEY:
    raise ValueError(
        "OLLAMA_API_KEY is missing. Please add OLLAMA_API_KEY in your .env file."
    )

client = Client(
    host=OLLAMA_HOST,
    headers={
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
)


def chat_completion(messages, temperature=0.2, num_predict=None):
    options = {
        "temperature": temperature
    }

    if num_predict is not None:
        options["num_predict"] = num_predict

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=options
    )

    return response["message"]["content"].strip()