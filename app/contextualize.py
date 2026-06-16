from pathlib import Path
from datetime import datetime
from ollama_config import chat_completion


CHAT_HISTORY_FILE = Path(__file__).parent / "chat_history.txt"


contextualize_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def save_message_to_file(role, content):
    """
    Saves user and assistant messages into a text file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {role}: {content}\n")


def read_last_10_messages_from_file():
    """
    Reads last 10 messages from chat_history.txt.
    """
    if not CHAT_HISTORY_FILE.exists():
        return ""

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
        lines = file.readlines()

    return "".join(lines[-10:])


def contextualize_query(query, messages=None):
    """
    Uses last 10 messages from text file as context and rewrites
    the latest user query into a standalone query.
    """
    history = read_last_10_messages_from_file()

    if not history.strip():
        save_message_to_file("user", query)
        return query

    prompt = f"""{contextualize_prompt}

Chat History:
{history}

Latest Question: {query}

Standalone Question:"""

    reformulated = chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    save_message_to_file("user", query)

    return reformulated