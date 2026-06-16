import sqlite3
import pandas as pd
from pathlib import Path
import re
from ollama_config import chat_completion


db_path = Path(__file__).parent / "db.sqlite"


sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags. 
<schema> 
table: product 

fields: 
product_link - string (hyperlink to product)	
title - string (name of the product)	
brand - string (brand of the product)	
price - integer (price of the product in Indian Rupees)	
discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)	
avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)	
total_ratings - integer (total number of ratings for the product)

</schema>

Rules:
- Generate only one SQLite SELECT query.
- The query must use SELECT *.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
- For brand/title search, use LOWER(column_name) LIKE LOWER('%value%').
- Never use ILIKE because SQLite does not support it.
- Always provide SQL between <SQL></SQL> tags.
- Do not write explanation outside the SQL tags.

Example:
<SQL>SELECT * FROM product WHERE LOWER(brand) LIKE LOWER('%nike%') LIMIT 5;</SQL>"""


comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided.

You will be provided with Question: and Data:.

Reply based only on the provided Data.
Do not say "Based on the data".
Do not make up products.
If Data is empty, say "No matching products found."

When asked about products, always reply in this format:

1. Product title: Rs. price (discount percent off), Rating: avg_rating, Link: product_link
2. Product title: Rs. price (discount percent off), Rating: avg_rating, Link: product_link

Use list format, one product per line."""


def generate_sql_query(question):
    """
    Takes a natural language question and uses Ollama Cloud
    to generate a corresponding SQLite query.
    """
    sql_response = chat_completion(
        messages=[
            {
                "role": "system",
                "content": sql_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0.2,
        num_predict=1024
    )

    return sql_response


def run_query(query):
    """
    Executes a SELECT query against SQLite and returns a DataFrame.
    """
    clean_query = query.strip().rstrip(";")

    blocked_words = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "PRAGMA",
        "TRUNCATE",
        "REPLACE"
    ]

    upper_query = clean_query.upper()

    if not upper_query.startswith("SELECT"):
        return None

    if any(word in upper_query for word in blocked_words):
        return None

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(clean_query, conn)
        return df


def data_comprehension(question, context):
    """
    Uses Ollama Cloud to generate a natural language response.
    """
    context = str(context)

    if len(context) > 6000:
        context = context[:6000]

    answer = chat_completion(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"Question: {question}\nDATA: {context}"
            }
        ],
        temperature=0.2
    )

    return answer


def extract_sql(sql_query):
    """
    Extracts SQL from <SQL></SQL> tags.
    """
    pattern = r"<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL | re.IGNORECASE)

    if not matches:
        return None

    return matches[0].strip()


def sql_chain(question):
    """
    Main text-to-SQL chain.
    """
    sql_query = generate_sql_query(question)
    query = extract_sql(sql_query)

    if not query:
        return "Sorry, LLM is not able to generate a query for this question."

    print("Generated SQL:", query)

    try:
        response = run_query(query)
    except Exception as e:
        print("SQL execution error:", str(e))
        return "Sorry, there was a problem executing the SQL query."

    if response is None:
        return "Sorry, only safe SELECT queries are allowed."

    if response.empty:
        return "No matching products found."

    context = response.to_dict(orient="records")

    answer = data_comprehension(question, context)
    return answer


if __name__ == "__main__":
    test_queries = [
        "Show top 3 shoes with best rating",
        "Show me 5 cheapest shoes",
        "Find Nike shoes",
        "Show products under 2000 rupees",
        "Show products with discount greater than 30 percent",
    ]

    for q in test_queries:
        print("\nQUERY:", q)
        print(sql_chain(q))