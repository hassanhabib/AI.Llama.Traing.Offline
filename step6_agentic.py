import pyodbc
from llama_cpp import Llama
import re

# ------------------------
# 🔍 Get DB schema
# ------------------------
def get_db_schema(cursor):
    schema = ""

    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG = DB_NAME()
    """)
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table}'
        """)
        columns = cursor.fetchall()
        schema += f"Table: {table}\n"
        for column_name, data_type in columns:
            schema += f"- {column_name} ({data_type})\n"
        schema += "\n"
    return schema.strip()

# ------------------------
# 🧠 Build LLM prompt
# ------------------------
def build_prompt(user_question, schema, error_message=None, previous_sql=None):
    if error_message:
        return f"""
You previously generated this SQL which failed:

{previous_sql}

The error was:
{error_message}

Try again. ONLY use the tables and columns listed in this schema.

Schema:
{schema}

User question:
"{user_question}"

Respond ONLY with a valid Microsoft SQL Server SELECT query and end it with a semicolon.
"""
    else:
        return f"""
You are a SQL expert. You will receive a database schema and a user question.

You MUST:
- Use ONLY table and column names exactly as provided in the schema
- NEVER invent or singularize table names like 'Student' if only 'Students' exists
- Output ONLY a valid Microsoft SQL Server SELECT statement ending with a semicolon

Schema:
{schema}

Question: "{user_question}"

Output:
"""

# ------------------------
# 🧼 Extract SELECT statement
# ------------------------
def extract_valid_sql(text):
    match = re.search(r"(SELECT\s.+?;)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("No valid SELECT statement found.")
    sql = match.group(1).strip()
    if "LIMIT" in sql.upper():
        raise ValueError("Invalid keyword 'LIMIT' for T-SQL.")
    return sql

# ------------------------
# 🚀 MAIN SCRIPT
# ------------------------

# Step 0: Connect to MSSQL
conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=BIGB;DATABASE=SchoolDb;Trusted_Connection=yes;")
cursor = conn.cursor()

# Step 1: Load local LLM
llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Step 2: User question
user_question = "Find the names of all students who have a score higher than 150."

# Step 3: Introspect schema
schema = get_db_schema(cursor)

# Step 4: Attempt loop
MAX_RETRIES = 2
attempt = 0
error_message = None
sql_query = None
data = None

while attempt < MAX_RETRIES:
    prompt = build_prompt(user_question, schema, error_message, sql_query)
    response = llm(prompt=prompt, max_tokens=200)
    raw_response = response['choices'][0]['text'].strip()
    print(f"🔍 Attempt {attempt + 1} - LLM Response:\n{raw_response}")

    try:
        sql_query = extract_valid_sql(raw_response)
        print("✅ Extracted SQL:\n", sql_query)

        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, row)) for row in results]

        if not data:
            raise ValueError("Query ran but returned no results.")

        break  # ✅ Success

    except Exception as e:
        error_message = str(e)
        print("🚨 Error:", error_message)
        attempt += 1

# Step 5: Summarize or report failure
if data:
    summary_prompt = f"""
Here are the results of the query:\n{data}
Summarize this nicely for the user.
"""
    summary = llm(prompt=summary_prompt, max_tokens=200)['choices'][0]['text'].strip()
    print("📄 Summary:\n", summary)
else:
    print("❌ Failed after retries. Please rephrase your question.")
