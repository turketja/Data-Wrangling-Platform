#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p notebooks
mkdir -p ui
mkdir -p chatbot

# Create UI files
cat > ui/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
EOF

cat > ui/requirements.txt << 'EOF'
streamlit==1.32.0
pandas==2.2.0
requests==2.31.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.27
minio==7.2.0
EOF

# Create UI app.py
cat > ui/app.py << 'EOF'
import os
import streamlit as st
import pandas as pd
import requests
import json
from minio import Minio
import psycopg2
from psycopg2.extras import RealDictCursor
import io

# Configuration
JUPYTER_URL = os.environ.get("JUPYTER_URL", "http://jupyter:8888")
CHATBOT_URL = os.environ.get("CHATBOT_URL", "http://chatbot:8000")
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

# Create default bucket if it doesn't exist
try:
    if not minio_client.bucket_exists("data"):
        minio_client.make_bucket("data")
except Exception as e:
    st.error(f"MinIO Error: {e}")

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
        st.error(f"PostgreSQL Error: {e}")
        return None

# Set page config
st.set_page_config(
    page_title="Data Analysis Hub",
    page_icon="📊",
    layout="wide"
)

# Create sidebar
st.sidebar.title("Data Analysis Hub")
page = st.sidebar.radio("Navigate", ["Home", "Data Upload", "File Explorer", "Chat", "Data Analysis", "Notebooks"])

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'tables' not in st.session_state:
    st.session_state.tables = []

# Fetch PostgreSQL tables
def fetch_tables():
    conn = get_postgres_conn()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [table[0] for table in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            st.error(f"Error fetching tables: {e}")
    return []

# Home page
if page == "Home":
    st.title("Welcome to Data Analysis Hub")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick Stats")
        try:
            objects = list(minio_client.list_objects("data"))
            st.metric("Files Stored", len(objects))
            
            tables = fetch_tables()
            st.metric("Database Tables", len(tables))
            
            st.metric("Chat Messages", len(st.session_state.chat_history))
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    
    with col2:
        st.subheader("Getting Started")
        st.markdown("""
        1. **Upload Data**: Start by uploading your data files
        2. **Explore Files**: View and manage your uploaded files
        3. **Chat with Data**: Ask questions about your data
        4. **Analyze Data**: Run analysis on your datasets
        5. **Notebooks**: Access Jupyter notebooks for advanced analysis
        """)

# Data Upload page
elif page == "Data Upload":
    st.title("Upload Data")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json", "txt"])
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Save to MinIO
        if st.button("Save to Storage"):
            try:
                minio_client.put_object(
                    "data", 
                    uploaded_file.name, 
                    uploaded_file, 
                    length=uploaded_file.size,
                    content_type=uploaded_file.type
                )
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                
                # If it's a CSV, offer to save to PostgreSQL
                if uploaded_file.name.endswith('.csv'):
                    if st.button("Also save to database"):
                        df = pd.read_csv(uploaded_file)
                        conn = get_postgres_conn()
                        if conn:
                            table_name = uploaded_file.name.split('.')[0].lower().replace(' ', '_')
                            df.to_sql(table_name, conn, if_exists='replace', index=False)
                            st.success(f"Data saved to table '{table_name}'")
                            conn.close()
                
            except Exception as e:
                st.error(f"Error: {e}")

# File Explorer page
elif page == "File Explorer":
    st.title("File Explorer")
    
    try:
        objects = list(minio_client.list_objects("data"))
        if not objects:
            st.info("No files found. Upload some files first!")
        else:
            file_names = [obj.object_name for obj in objects]
            selected_file = st.selectbox("Select a file", file_names)
            
            if selected_file:
                st.session_state.current_file = selected_file
                
                # Get file info
                file_info = minio_client.stat_object("data", selected_file)
                st.write(f"Size: {file_info.size} bytes | Last modified: {file_info.last_modified}")
                
                # Preview file
                st.subheader("File Preview")
                
                try:
                    response = minio_client.get_object("data", selected_file)
                    if selected_file.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(response.read()))
                        st.dataframe(df.head(10))
                        
                        # Option to load into PostgreSQL
                        if st.button("Load into database"):
                            conn = get_postgres_conn()
                            if conn:
                                table_name = selected_file.split('.')[0].lower().replace(' ', '_')
                                # Re-read the file since we already read it above
                                response = minio_client.get_object("data", selected_file)
                                df = pd.read_csv(io.BytesIO(response.read()))
                                df.to_sql(table_name, conn, if_exists='replace', index=False)
                                st.success(f"Data loaded to table '{table_name}'")
                                conn.close()
                                
                    elif selected_file.endswith('.xlsx'):
                        df = pd.read_excel(io.BytesIO(response.read()))
                        st.dataframe(df.head(10))
                    elif selected_file.endswith('.json'):
                        data = json.loads(response.read().decode('utf-8'))
                        st.json(data)
                    else:
                        content = response.read().decode('utf-8')
                        st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
                        
                except Exception as e:
                    st.error(f"Error previewing file: {e}")
                    
    except Exception as e:
        st.error(f"Error accessing storage: {e}")

# Chat page
elif page == "Chat":
    st.title("Chat with Your Data")
    
    # Sidebar for context selection
    st.sidebar.subheader("Chat Context")
    context_type = st.sidebar.radio("Select context", ["File", "Database Table"])
    
    if context_type == "File":
        try:
            objects = list(minio_client.list_objects("data"))
            file_names = [obj.object_name for obj in objects]
            selected_context = st.sidebar.selectbox("Select a file for context", ["None"] + file_names)
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}")
            selected_context = "None"
    else:
        tables = fetch_tables()
        selected_context = st.sidebar.selectbox("Select a table for context", ["None"] + tables)
    
    # Display chat history
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
    
    # Chat input
    user_input = st.text_input("Ask a question about your data:")
    
    if st.button("Send") and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Send to chatbot API
        try:
            context_data = None
            
            # Get context data if selected
            if selected_context != "None":
                if context_type == "File":
                    response = minio_client.get_object("data", selected_context)
                    context_data = response.read().decode('utf-8')
                else:
                    conn = get_postgres_conn()
                    if conn:
                        cursor = conn.cursor(cursor_factory=RealDictCursor)
                        cursor.execute(f"SELECT * FROM {selected_context} LIMIT 100")
                        context_data = json.dumps(cursor.fetchall())
                        conn.close()
            
            # Call chatbot API
            chatbot_response = requests.post(
                f"{CHATBOT_URL}/chat",
                json={
                    "message": user_input,
                    "history": st.session_state.chat_history,
                    "context": context_data,
                    "context_type": context_type.lower(),
                    "context_name": selected_context
                }
            )
            
            if chatbot_response.status_code == 200:
                response_data = chatbot_response.json()
                st.session_state.chat_history.append({"role": "assistant", "content": response_data["response"]})
            else:
                st.error(f"Error from chatbot API: {chatbot_response.text}")
                st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {str(e)}"})
        
        # Force refresh
        st.experimental_rerun()

# Data Analysis page
elif page == "Data Analysis":
    st.title("Data Analysis")
    
    tab1, tab2 = st.tabs(["Database Tables", "Files"])
    
    with tab1:
        tables = fetch_tables()
        if not tables:
            st.info("No tables found in the database.")
        else:
            selected_table = st.selectbox("Select a table", tables)
            
            if selected_table:
                conn = get_postgres_conn()
                if conn:
                    try:
                        # Load data
                        df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
                        st.write(f"Table has {len(df)} rows and {len(df.columns)} columns")
                        
                        # Display data
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Basic statistics
                        st.subheader("Basic Statistics")
                        st.write(df.describe())
                        
                        # Column selection for visualization
                        st.subheader("Visualization")
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        if numeric_cols:
                            x_col = st.selectbox("Select X axis", df.columns)
                            y_col = st.selectbox("Select Y axis", numeric_cols)
                            
                            chart_type = st.radio("Select chart type", ["Bar", "Line", "Scatter"])
                            
                            if chart_type == "Bar":
                                st.bar_chart(df.set_index(x_col)[y_col])
                            elif chart_type == "Line":
                                st.line_chart(df.set_index(x_col)[y_col])
                            else:
                                st.scatter_chart(df, x=x_col, y=y_col)
                        else:
                            st.info("No numeric columns available for visualization")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        conn.close()
    
    with tab2:
        try:
            objects = list(minio_client.list_objects("data"))
            file_names = [obj.object_name for obj in objects if obj.object_name.endswith(('.csv', '.xlsx'))]
            
            if not file_names:
                st.info("No CSV or Excel files found for analysis.")
            else:
                selected_file = st.selectbox("Select a file", file_names)
                
                if selected_file:
                    # Load the file
                    response = minio_client.get_object("data", selected_file)
                    
                    if selected_file.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(response.read()))
                    else:  # xlsx
                        df = pd.read_excel(io.BytesIO(response.read()))
                    
                    st.write(f"File has {len(df)} rows and {len(df.columns)} columns")
                    
                    # Display data
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Basic statistics
                    st.subheader("Basic Statistics")
                    st.write(df.describe())
                    
                    # Visualization
                    st.subheader("Visualization")
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        x_col = st.selectbox("Select X axis", df.columns)
                        y_col = st.selectbox("Select Y axis", numeric_cols)
                        
                        chart_type = st.radio("Select chart type", ["Bar", "Line", "Scatter"])
                        
                        if chart_type == "Bar":
                            st.bar_chart(df.set_index(x_col)[y_col])
                        elif chart_type == "Line":
                            st.line_chart(df.set_index(x_col)[y_col])
                        else:
                            st.scatter_chart(df, x=x_col, y=y_col)
                    else:
                        st.info("No numeric columns available for visualization")
                        
        except Exception as e:
            st.error(f"Error: {e}")

# Notebooks page
elif page == "Notebooks":
    st.title("Jupyter Notebooks")
    
    st.markdown(f"""
    Access JupyterLab for advanced data analysis by clicking the button below:
    
    <a href="{JUPYTER_URL}" target="_blank">
        <button style="background-color:#4CAF50;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
            Open JupyterLab
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Working with Data in Notebooks
    
    Your uploaded files are available in the `/data` directory within JupyterLab.
    
    #### Example code to load data:
    
    ```python
    import pandas as pd
    
    # Load CSV file
    df = pd.read_csv('/home/jovyan/data/your_file.csv')
    
    # Or connect to PostgreSQL
    from sqlalchemy import create_engine
    
    engine = create_engine('postgresql://datauser:datapass@postgres:5432/datadb')
    df = pd.read_sql('SELECT * FROM your_table', engine)
    ```
    """)
EOF

# Create Chatbot files
cat > chatbot/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > chatbot/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
pandas==2.2.0
psycopg2-binary==2.9.9
minio==7.2.0
EOF

# Create Chatbot app.py
cat > chatbot/app.py << 'EOF'
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
EOF

# Write docker-compose.yml if it doesn't exist
if [ ! -f "docker-compose.yml" ]; then
  cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Main UI - streamlit app that provides unified access
  ui:
    build:
      context: ./ui
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data
      - ./ui:/app
    environment:
      - JUPYTER_URL=http://jupyter:8888
      - POSTGRES_USER=datauser
      - POSTGRES_PASSWORD=datapass
      - POSTGRES_DB=datadb
      - POSTGRES_HOST=postgres
      - CHATBOT_URL=http://chatbot:8000
      - MINIO_URL=http://minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    depends_on:
      - jupyter
      - postgres
      - chatbot
      - minio

  # JupyterLab for interactive data analysis
  jupyter:
    image: jupyter/datascience-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./data:/home/jovyan/data
      - ./notebooks:/home/jovyan/notebooks
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: >
      start-notebook.sh 
      --NotebookApp.token='' 
      --NotebookApp.password='' 
      --NotebookApp.allow_origin='*'

  # PostgreSQL for structured data storage
  postgres:
    image: postgres:14
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=datauser
      - POSTGRES_PASSWORD=datapass
      - POSTGRES_DB=datadb

  # MinIO for S3-compatible object storage
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"

  # Chatbot service for querying data
  chatbot:
    build:
      context: ./chatbot
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    environment:
      - POSTGRES_USER=datauser
      - POSTGRES_PASSWORD=datapass
      - POSTGRES_DB=datadb
      - POSTGRES_HOST=postgres
      - MINIO_URL=http://minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin

volumes:
  postgres-data:
  minio-data:
EOF
fi

echo "Starting containers..."
docker-compose up -d

echo "Waiting for services to be available..."
sleep 10

echo "Setup complete! Access your data analysis platform at:"
echo "  - Main UI: http://localhost:8501"
echo "  - JupyterLab: http://localhost:8888"
echo "  - MinIO: http://localhost:9001"
echo ""
echo "The system is now ready to use. You can upload data files and start analyzing them."
