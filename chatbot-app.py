from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import io
from minio import Minio

app = FastAPI()

# Environment variables
MINIO_URL = os.environ.get("MINIO_URL", "http://minio:9000").replace("http://", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "datauser")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "datapass")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "datadb")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")

# Initialize MinIO client
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Initialize PostgreSQL connection
def get_postgres_conn():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"PostgreSQL Error: {e}")
        return None

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    context: Optional[str] = None
    context_type: Optional[str] = None
    context_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

# Helper functions for data analysis
def analyze_csv_data(data_str, query):
    """Analyze CSV data based on the query."""
    try:
        df = pd.read_csv(io.StringIO(data_str))
        return analyze_dataframe(df, query)
    except Exception as e:
        return f"Error analyzing CSV data: {str(e)}"

def analyze_table_data(table_name, query):
    """Analyze database table based on the query."""
    try:
        conn = get_postgres_conn()
        if not conn:
            return "Could not connect to the database."
        
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return analyze_dataframe(df, query)
    except Exception as e:
        return f"Error analyzing table data: {str(e)}"

def analyze_dataframe(df, query):
    """Analyze pandas DataFrame based on the query."""
    try:
        # Basic info about the data
        if any(keyword in query.lower() for keyword in ["describe", "summary", "statistics", "stats"]):
            return f"Here are the summary statistics:\n{df.describe().to_string()}"
        
        # Column info
        elif any(keyword in query.lower() for keyword in ["columns", "fields"]):
            return f"The dataset has the following columns: {', '.join(df.columns.tolist())}"
        
        # Count rows
        elif any(keyword in query.lower() for keyword in ["count", "how many rows", "size"]):
            return f"The dataset has {len(df)} rows and {len(df.columns)} columns."
        
        # Check for missing values
        elif any(keyword in query.lower() for keyword in ["missing", "null", "na"]):
            missing = df.isnull().sum()
            return f"Missing values per column:\n{missing.to_string()}"
        
        # Get max/min values
        elif "maximum" in query.lower() or "max" in query.lower():
            for col in df.select_dtypes(include=['number']).columns:
                if col.lower() in query.lower():
                    return f"The maximum value in {col} is {df[col].max()}"
            return f"Numeric column maximums:\n{df.max().to_string()}"
        
        elif "minimum" in query.lower() or "min" in query.lower():
            for col in df.select_dtypes(include=['number']).columns:
                if col.lower() in query.lower():
                    return f"The minimum value in {col} is {df[col].min()}"
            return f"Numeric column minimums:\n{df.min().to_string()}"
        
        # Get mean/average values
        elif any(keyword in query.lower() for keyword in ["mean", "average"]):
            for col in df.select_dtypes(include=['number']).columns:
                if col.lower() in query.lower():
                    return f"The mean value of {col} is {df[col].mean()}"
            return f"Numeric column means:\n{df.mean().to_string()}"
        
        # If no specific analysis is detected
        else:
            return (
                f"I can analyze this dataset with {len(df)} rows and {len(df.columns)} columns. "
                f"You can ask about statistics, columns, missing values, maximums, minimums, or averages."
            )
            
    except Exception as e:
        return f"Error during analysis: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """Process a chat request with optional data context."""
    try:
        query = request.message
        
        # If we have context data, analyze it
        if request.context and request.context_name != "None":
            if request.context_type == "file":
                response_text = analyze_csv_data(request.context, query)
            elif request.context_type == "database table":
                response_text = analyze_table_data(request.context_name, query)
            else:
                response_text = "I'm not sure how to analyze this type of data context."
        else:
            # General chat without data context
            response_text = (
                "I'm your data assistant. To analyze data, please select a file or database table as context. "
                "Then you can ask questions about the data such as statistics, columns, missing values, etc."
            )
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
