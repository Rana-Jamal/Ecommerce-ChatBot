import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from ollama_config import chat_completion


faqs_path = Path(__file__).parent / "resources/faq_data.csv"

chroma_client = chromadb.Client()
collection_name_faq = "faqs"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def ingest_faq_data(path):
    """
    Reads the FAQ data from the CSV file and ingests it into a ChromaDB collection.
    It only creates the collection if it doesn't already exist.
    """
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:

        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function=ef,
        )

        df = pd.read_csv(path)
        docs = df["question"].to_list()

        collection.add(
            documents=docs,
            metadatas=[{"answer": ans} for ans in df["answer"].to_list()],
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

    result = collection.query(
        query_texts=[query],
        n_results=2,
    )

    return result


def faq_chain(query):
    """
    Main chain that handles FAQ queries.
    """
    result = get_relevant_qa(query)

    context = "\n".join(
        [r.get("answer", "") for r in result["metadatas"][0]]
    )

    answer = generate_answer(query, context)
    return answer


def generate_answer(query, context):
    """
    Uses Ollama Cloud to generate an answer based strictly on the provided context.
    """
    prompt = f"""Given the question and context below, generate the answer based on the context only.
If you don't find the answer inside the context then say "I don't know the answer." Do not make things up.

QUESTION: {query}
CONTEXT: {context}"""

    answer = chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return answer


if __name__ == "__main__":
    ingest_faq_data(faqs_path)

    query = "Do you take cash as a payment option"

    answer = faq_chain(query)
    print(answer)