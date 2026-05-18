import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables, including the Groq API key
load_dotenv()

# Initialize the Groq client for making API calls
client = Groq()

# Retrieve the specific Groq model to use from environment variables
groq_model = os.getenv("GROQ_MODEL")

# The system prompt that instructs the LLM on how to reformulate the query
contextualize_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

def contextualize_query(query, messages):
    """
    Takes the current user query and the chat history, and uses an LLM
    to rewrite the query so that it makes sense on its own.
    """
    # If there is no chat history, return the query as is
    if not messages:
        return query
    
    # Extract the last 6 messages to form the recent chat history
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-6:]])
    
    # Combine the instructions, chat history, and the new query
    prompt = f"{contextualize_prompt}\n\nChat History:\n{history}\n\nLatest Question: {query}\n\nStandalone Question:"
    
    # Call the Groq API to generate the reformulated question
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=groq_model,
        temperature=0.1
    )
    
    # Extract and return the reformulated question from the API response
    reformulated = chat_completion.choices[0].message.content.strip()
    return reformulated
