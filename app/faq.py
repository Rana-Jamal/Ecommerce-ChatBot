import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client for LLM calls
groq_client = Groq()
groq_model = os.getenv("GROQ_MODEL")

# Path to the CSV file containing the FAQ data
faqs_path = Path(__file__).parent / "resources/faq_data.csv"

# Initialize ChromaDB client for vector storage
chroma_client = chromadb.Client()
collection_name_faq = "faqs"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)


def ingest_faq_data(path):
    """
    Reads the FAQ data from the CSV file and ingests it into a ChromaDB collection.
    It only creates the collection if it doesn't already exist.
    """
    # Check if the FAQ collection already exists in ChromaDB
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        
        # Create a new collection with the specified embedding function
        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function=ef,
        )
        
        # Read the CSV data using pandas
        df = pd.read_csv(path)
        docs = df['question'].to_list()

        collection.add(
            documents=docs,
            metadatas=[{'answer': ans} for ans in df['answer'].to_list()],
            ids=[f"id_{i}" for i in range(len(docs))]
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")

    else:
        print(f"Collection {collection_name_faq} already exists")

def get_relevant_qa(query):
    """
    Queries the ChromaDB collection to find the most relevant FAQs based on the user query.
    """
    collection = chroma_client.get_collection(name=collection_name_faq)
    
    # Retrieve the top 2 most relevant results
    result = collection.query(
        query_texts=[query],
        n_results=2,
    )
    return result


def faq_chain(query):
    """
    Main chain that handles FAQ queries: it retrieves relevant context from the database
    and generates an answer using the LLM.
    """
    # Get relevant questions and answers from ChromaDB
    result = get_relevant_qa(query)
    
    # Extract the answers from the metadata to form the context
    context = ''.join([r.get('answer') for r in result['metadatas'][0]])
    
    # Generate a final answer using the LLM
    answer = generate_answer(query, context)
    return answer


def generate_answer(query, context):
    """
    Uses the Groq LLM to generate an answer to the user's query based strictly on the provided context.
    """
    prompt = f'''Given the question and context below, generate the answer based on the context only.
        If you don't find the answer inside the context then say "I don't know the answer." Do not make things up.
        
        QUESTION: {query}
        CONTEXT: {context}'''

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=groq_model
    )
    
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    ingest_faq_data(faqs_path)

    query = "Do you take cash as a payment option"

    answer = faq_chain(query)
    print(answer)
