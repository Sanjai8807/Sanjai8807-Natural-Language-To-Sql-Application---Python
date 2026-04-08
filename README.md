# 🧠 Natural Language to SQL App

A Streamlit-based web application that converts plain English questions into MySQL queries using **Google Gemini** as the LLM and **LangChain** as the orchestration layer.

---

## 📸 Features

- 🔍 Ask questions in plain English — get SQL results instantly
- 🤖 Powered by `gemini-2.5-flash-lite` via Google Generative AI
- 🛡️ Only `SELECT` queries allowed (safe read-only access)
- 📊 Smart result display — single aggregates shown as metrics, tabular data as dataframes
- 🔢 Handles `Decimal` types from MySQL gracefully

---

## 🗂️ Project Structure

```
nl-to-sql-app/
├── app.py               # Main Streamlit application
├── .env                 # Environment variables (not committed)
├── requirements.txt     # Python dependencies
└── README.md
```
<img width="1189" height="813" alt="image" src="https://github.com/user-attachments/assets/fb4254c1-5b0a-4106-8ae6-5a3c481cafe6" />

<img width="1228" height="618" alt="image" src="https://github.com/user-attachments/assets/cae60cdd-12da-4f83-8844-9da04b96f385" />

<img width="1213" height="719" alt="image" src="https://github.com/user-attachments/assets/1b1b5dec-47d6-492b-9580-6a319f2ba398" />

<img width="1212" height="792" alt="image" src="https://github.com/user-attachments/assets/1b8c2aff-a687-432a-b638-46a56e89494a" />

<img width="1163" height="821" alt="image" src="https://github.com/user-attachments/assets/31a46585-ba97-4077-b702-95e702eb6a02" />

---

## ⚙️ Prerequisites

- Python 3.9+
- MySQL running locally on port `3306`
- A database named `retail_sales_db`
- A Google AI API key ([get one here](https://aistudio.google.com/app/apikey))

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nl-to-sql-app.git
cd nl-to-sql-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Set up the MySQL database

Make sure your MySQL server is running and the `retail_sales_db` database exists. Update the connection settings in `app.py` if your credentials differ:

```python
connection_url = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password="root123",
    host="localhost",
    port=3306,
    database="retail_sales_db",
)
```

### 6. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```txt
streamlit
langchain
langchain-google-genai
langchain-community
sqlalchemy
pymysql
pandas
python-dotenv
google-generativeai
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📄 Application Code (`app.py`)

```python
import os
import ast
import pandas as pd
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from decimal import Decimal

# Load environment variables
load_dotenv()

# -------------------- PROMPT --------------------
template = """
You are a MySQL expert.

Given the following database schema:
{table_info}

Write a syntactically correct MySQL query.

STRICT RULES:
- Output ONLY raw SQL
- NO markdown (no sql)
- NO explanations
- NO "SQLQuery:" prefix
- Ensure proper spacing
- Limit results to {top_k} rows unless specified

Question: {input}
"""

prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=template,
)

# -------------------- DB CONFIG --------------------
connection_url = URL.create(
    drivername="mysql+pymysql",
    username="xxxx",
    password="xxxx",
    host="localhost",
    port=3306,
    database="retail_sales_db",
)

engine = create_engine(connection_url)
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# -------------------- LLM --------------------
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# -------------------- CHAIN --------------------
chain = create_sql_query_chain(llm, db, prompt=prompt)

# -------------------- CLEAN SQL --------------------
def clean_sql(query: str) -> str:
    query = query.strip()
    query = query.replace("sql", "").replace("```", "").strip()
    if query.startswith("SQLQuery:"):
        query = query.replace("SQLQuery:", "").strip()
    if query.lower().startswith("sql"):
        query = query[3:].strip()
    return query

# -------------------- FORMAT RESULT --------------------
def display_result(question: str, rows: list, columns: list):
    """Display query results — rows is a real list of tuples, columns is a list of col names."""
    if not rows:
        st.info("Query returned no results.")
        return

    df = pd.DataFrame(rows, columns=columns)

    # Convert Decimal → float for all numeric columns
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, Decimal)).any():
            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)

    # Single aggregate value (SUM, COUNT, AVG, etc.)
    if df.shape == (1, 1):
        value = df.iloc[0, 0]
        try:
            st.metric(label=question, value=f"{float(value):,.2f}")
        except (TypeError, ValueError):
            st.metric(label=question, value=str(value))
    else:
        st.dataframe(df, use_container_width=True)

# -------------------- EXECUTION --------------------
def execute_query(question: str):
    try:
        response = chain.invoke({
            "question": question,
            "input": question,
            "top_k": 5,
        })

        cleaned_query = clean_sql(response)
        print("Final SQL:", cleaned_query)

        if not cleaned_query.lower().startswith("select"):
            st.error("Only SELECT queries are allowed.")
            return None, None, None

        # Run query directly via SQLAlchemy — get real rows + column names
        with engine.connect() as conn:
            result = conn.execute(text(cleaned_query))
            columns = list(result.keys())
            rows = result.fetchall()

        return cleaned_query, rows, columns

    except ProgrammingError as e:
        st.error(f"SQL Error: {e}")
        return None, None, None

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return None, None, None

# -------------------- STREAMLIT UI --------------------
st.title("🧠 Natural Language to SQL App")

question = st.text_input("Enter your question:")

if st.button("Execute"):
    if question:
        with st.spinner("Generating SQL and fetching results..."):
            cleaned_query, rows, columns = execute_query(question)

        if cleaned_query and rows is not None:
            st.subheader("Generated SQL Query:")
            st.code(cleaned_query, language="sql")

            st.subheader("Query Result:")
            display_result(question, rows, columns)
        else:
            st.warning("No result returned.")
    else:
        st.warning("Please enter a question.")
```

---

## 💡 Example Questions

| Question | What it does |
|---|---|
| `What is the total revenue?` | Returns a single metric (SUM) |
| `Show the top 5 customers by sales` | Returns a table |
| `How many orders were placed in January?` | Returns a COUNT metric |
| `List all products with price above 500` | Returns a filtered table |

---

## 🧩 How It Works

```
User Question
     │
     ▼
LangChain SQL Chain  ←  DB Schema (table_info)
     │
     ▼
Google Gemini LLM  (gemini-2.5-flash-lite)
     │
     ▼
Raw SQL (cleaned & validated)
     │
     ▼
SQLAlchemy executes on MySQL
     │
     ▼
Results → Streamlit UI (metric or dataframe)
```

1. The user types a natural language question.
2. LangChain passes the question + DB schema to Gemini.
3. Gemini returns a raw SQL query.
4. The `clean_sql()` function strips any markdown fences or prefixes.
5. Only `SELECT` queries are executed for safety.
6. Results are displayed as a metric (for single aggregate values) or a dataframe (for tabular data).

---

## 🔒 Security Notes

- Only `SELECT` queries are permitted — insert, update, delete, and drop statements are blocked.
- Keep your `.env` file out of version control. Add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

- Avoid exposing this app publicly without adding authentication.

---

## 🛠️ Customization

**Change the LLM model:** Edit the `model` parameter in `app.py`:
```python
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   # swap model here
    ...
)
```

