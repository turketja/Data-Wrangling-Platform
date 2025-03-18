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
