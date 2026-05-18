import streamlit as st
from router import router
from contextualize import contextualize_query

from faq import ingest_faq_data, faqs_path, faq_chain
from sql import sql_chain
from smalltalk import talk

# Set the title of the Streamlit application
st.title("E-commerce chatbot")

# Load and ingest the FAQ data into the knowledge base
ingest_faq_data(faqs_path)

def ask(query):
    """
    Route the user's query to the appropriate handler 
    based on the semantic router's prediction.
    """
    if router(query).name == 'faq':
        return faq_chain(query)
    elif router(query).name == 'sql':
        return sql_chain(query)
    elif router(query).name == 'small-talk':
        return talk(query)
    else:
        return "Sorry, route not implemented yet."

# Get the user input from the chat interface
query = st.chat_input("Write a query")

# Initialize the chat history in the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display all previous messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If the user has submitted a new query
if query:
    # Display the user's query in the chat
    with st.chat_message("user"):
        st.markdown(query)
    
    # Reformulate the query based on the chat history so it can be understood standalone
    contextualized_q = contextualize_query(query, st.session_state.messages)
    print(f"Original Query: {query}")
    print(f"Contextualized Query: {contextualized_q}")

    # Add the user's original query to the chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get the response from the appropriate routing chain
    response = ask(contextualized_q)
    
    # Display the assistant's response in the chat
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

