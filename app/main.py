import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from router import router
from contextualize import contextualize_query, save_message_to_file

from faq import ingest_faq_data, faqs_path, faq_chain
from sql import sql_chain
from smalltalk import talk


st.title("E-commerce chatbot")

ingest_faq_data(faqs_path)


def ask(query):
    """
    Route the user's query to the appropriate handler.
    """
    route_result = router(query)
    route_name = route_result.name if route_result else None

    print(f"Route: {route_name}")

    if route_name == "faq":
        return faq_chain(query)
    elif route_name == "sql":
        return sql_chain(query)
    elif route_name == "small-talk":
        return talk(query)
    else:
        return "Sorry, route not implemented yet."


query = st.chat_input("Write a query")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query:
    with st.chat_message("user"):
        st.markdown(query)

    contextualized_q = contextualize_query(query, st.session_state.messages)

    print(f"Original Query: {query}")
    print(f"Contextualized Query: {contextualized_q}")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    response = ask(contextualized_q)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    save_message_to_file("assistant", response)