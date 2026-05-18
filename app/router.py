from semantic_router import Route, SemanticRouter

import os
from dotenv import load_dotenv

from semantic_router.encoders import HuggingFaceEncoder

# Initialize the embedding model used for semantic routing
encoder = HuggingFaceEncoder(
    name= 'sentence-transformers/all-MiniLM-L6-v2'
)

# Load environment variables (like API keys) from a .env file
load_dotenv()

# Define a route for Frequently Asked Questions (FAQs)
faq = Route(
    name="faq",
    utterances=[
        "What is your return policy?",
        "How long does shipping take?",
        "Do you offer international delivery?",
        "What payment methods do you accept?",
        "How can I track my order?",
        "Can I cancel my order after placing it?",
        "Do you have a customer loyalty program?",
        "How do I contact customer support?",
        "What are your store hours?",
        "Do you offer gift wrapping services?",
    ],
    score_threshold=0.3,
)

# Define a route for database/SQL related queries (e.g., product searches)
# This could be used as an indicator to our chatbot to switch to a more
# conversational prompt or text-to-SQL pipeline
sql = Route(
    name="sql",
    utterances=[
        "Show me all laptops under $1000",
        "What colors are available for the iPhone 15?",
        "Find running shoes in size 10",
        "Which products are currently on sale?",
        "Show me the specifications of the Samsung Galaxy S24",
        "What's the price of the Sony WH-1000XM5 headphones?",
        "Do you have any leather jackets in stock?",
        "Find all products from the brand Nike",
        "What's the average rating of the MacBook Air?",
        "Show me wireless earbuds with noise cancellation",
    ],
    score_threshold=0.3,
)

# Define a route for general chit-chat or small talk
small_talk = Route(
    name= "small-talk",
    utterances=[
        "How are you?",
        "What is your name?",
        "Are you a robot?",
        "What are you?",
        "What do you do?",
    ],
    score_threshold=0.1,
)

# Combine the routes and the encoder into a SemanticRouter instance
router = SemanticRouter(routes= [faq, sql, small_talk], encoder= encoder, auto_sync="local" )


if __name__ == "__main__":

    # Test queries to see how they are routed
    query1 = "what is the  price of nike air jorder"
    query2 = "Do you accept cash payment?"

    # Evaluate the test queries
    result1 = router(query1)
    result2 = router(query2)

    # Print the predicted route name and its similarity score
    print(f" query1 : {result1.name , result1.similarity_score}")
    print(f" query2 : {result2.name, result2.similarity_score}")
    print()
