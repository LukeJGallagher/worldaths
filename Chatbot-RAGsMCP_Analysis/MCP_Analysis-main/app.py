import os
import streamlit as st
import requests
from dotenv import load_dotenv
import pandas as pd
import polars as pl
import json
import subprocess
import time
import tempfile
from datetime import datetime
import uuid
import socket
import argparse
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import pandasai as pai
from PIL import Image

# Load environment variables
load_dotenv()

# Server configurations (configurable via environment variables or command line args)
GITHUB_SERVER_URL = os.getenv('GITHUB_SERVER_URL')
POSTGRES_SERVER_URL = os.getenv('POSTGRES_SERVER_URL')
MYSQL_SERVER_URL = os.getenv('MYSQL_SERVER_URL')

# Global variables to store server processes
github_server_process = None
postgres_server_process = None
mysql_server_process = None
temp_data_dir = os.path.join(os.getcwd(), "temp_data")

# Create temp directory if it doesn't exist
if not os.path.exists(temp_data_dir):
    os.makedirs(temp_data_dir)

# Initialize settings in session state if not already present
if 'use_polars' not in st.session_state:
    st.session_state.use_polars = True  # Default to using Polars for better performance
if 'analysis_backend' not in st.session_state:
    st.session_state.analysis_backend = "openrouter"  # Default to OpenRouter
if 'current_temp_file' not in st.session_state:
    st.session_state.current_temp_file = None  # Track the current temp file

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except socket.error:
                continue
    return None

def start_server(server_name):
    """Start a server process"""
    st.error("Starting servers locally is disabled when deploying to Streamlit Cloud. Please ensure the GITHUB_SERVER_URL and POSTGRES_SERVER_URL environment variables are set correctly.")
    return None

def stop_server(process):
    """Stop a server process"""
    st.error("Stopping servers locally is disabled when deploying to Streamlit Cloud.")
    return None

def check_server_status(url):
    """Check if a server is running"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        return False
    except:
        return False

def save_dataframe(df, description="query_results", reuse_current=False):
    """
    Save a DataFrame to a file in the temp directory
    If using Polars, saves as Parquet for better performance
    If using Pandas, saves as CSV
    
    When reuse_current=True, will overwrite the existing temp file if available
    Otherwise creates a new temp file and updates the current_temp_file reference
    """
    # If reuse_current=True and we have a current temp file, use that path
    if reuse_current and st.session_state.current_temp_file and os.path.exists(st.session_state.current_temp_file):
        filepath = st.session_state.current_temp_file
    else:
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        
        if st.session_state.use_polars:
            # Save as Parquet
            filename = f"{description}_{timestamp}_{unique_id}.parquet"
            filepath = os.path.join(temp_data_dir, filename)
        else:
            # Save as CSV
            filename = f"{description}_{timestamp}_{unique_id}.csv"
            filepath = os.path.join(temp_data_dir, filename)
        
        # Store the current temp file path
        st.session_state.current_temp_file = filepath
    
    # Save the dataframe with the appropriate format
    if st.session_state.use_polars:
        # Convert to Polars if we received a Pandas dataframe
        if isinstance(df, pd.DataFrame):
            pl_df = pl.from_pandas(df)
        else:
            pl_df = df
            
        # Save as Parquet
        pl_df.write_parquet(filepath)
    else:
        # Convert to pandas if we received a Polars dataframe
        if isinstance(df, pl.DataFrame):
            pd_df = df.to_pandas()
        else:
            pd_df = df
            
        # Save as CSV
        pd_df.to_csv(filepath, index=False)
    
    # Return the file path
    return filepath

def analyze_data_with_openrouter(df, question, sample_rows=None, max_rows=1000):
    """
    Use OpenRouter to analyze the DataFrame based on a question
    
    Parameters:
        df: DataFrame to analyze (pandas or polars)
        question: string question to ask about the data
        sample_rows: if set, only use this many sample rows (set to None to use full dataset)
        max_rows: maximum number of rows to send (to avoid extremely large payloads)
    """
    try:
        # Convert to the appropriate dataframe type based on the setting
        if st.session_state.use_polars:
            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)
        else:
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        
        # Get schema information
        if st.session_state.use_polars:
            schema_info = "\n".join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
        else:
            schema_info = df.dtypes.to_string()
            
        # For LLMs, a more structured representation might be better than just to_string()
        # Prepare a structured string representation of data in a format optimized for LLMs
        def dataframe_to_structured_string(df, rows_limit=None):
            if rows_limit is not None:
                if st.session_state.use_polars:
                    df_sample = df.head(rows_limit)
                else:
                    df_sample = df.head(rows_limit)
            else:
                df_sample = df
                
            # For polars
            if st.session_state.use_polars:
                header = " | ".join(str(col) for col in df_sample.columns)
                divider = "-" * len(header)
                rows = []
                
                # Convert to readable format
                for row in df_sample.iter_rows():
                    rows.append(" | ".join(str(val) for val in row))
                
                row_strings = "\n".join(rows)
                return f"{header}\n{divider}\n{row_strings}"
            else:
                # For pandas
                return df_sample.to_string(index=False)
        
        # Convert DataFrame to string representation
        # If sample_rows is None, use the full dataset up to max_rows
        if sample_rows is not None and len(df) > sample_rows:
            # Use just a sample of the data
            data_sample = dataframe_to_structured_string(df, sample_rows)
            data_description = f"Data Preview (first {sample_rows} rows of {len(df)} total):"
        else:
            # Use the full dataset (up to max_rows)
            rows_to_use = min(len(df), max_rows)
            if rows_to_use < len(df):
                data_sample = dataframe_to_structured_string(df, rows_to_use)
                data_description = f"Data (first {rows_to_use} rows of {len(df)} total - truncated for size):"
            else:
                data_sample = dataframe_to_structured_string(df)
                data_description = f"Complete Data ({len(df)} rows):"
        
        # Create a JSON request to the OpenRouter API
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://mcp-learning.example.com",
            "Content-Type": "application/json"
        }
        
        # Get basic statistics to help the LLM
        try:
            if st.session_state.use_polars:
                # Calculate basic stats with Polars
                numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                               if pl.datatypes.is_numeric(dtype)]
                
                if numeric_cols:
                    stats = df.select([
                        pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols
                    ] + [
                        pl.col(col).min().alias(f"{col}_min") for col in numeric_cols
                    ] + [
                        pl.col(col).max().alias(f"{col}_max") for col in numeric_cols
                    ] + [
                        pl.col(col).std().alias(f"{col}_std") for col in numeric_cols
                    ])
                    
                    stats_str = "Basic Statistics:\n"
                    for col in numeric_cols:
                        stats_str += f"{col}: mean={stats[0][f'{col}_mean']:.2f}, min={stats[0][f'{col}_min']}, "
                        stats_str += f"max={stats[0][f'{col}_max']}, std={stats[0][f'{col}_std']:.2f}\n"
                else:
                    stats_str = "No numeric columns for statistics."
            else:
                # Calculate basic stats with Pandas
                desc = df.describe().to_string()
                stats_str = f"Basic Statistics:\n{desc}"
        except Exception as e:
            stats_str = f"Error calculating statistics: {str(e)}"
        
        # Enhanced context for the LLM
        system_content = f"""You are a data analysis expert. Analyze the following data carefully and answer the user's question.

Data Schema:
{schema_info}

{data_description}
{data_sample}

{stats_str}

Please provide a thorough, insightful analysis that addresses the user's question directly. 
Include relevant patterns, anomalies, and insights that would be valuable. 
When appropriate, suggest further analyses that could yield additional insights."""

        data = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3
        }
        
        # Make the API request
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60  # Increased timeout for larger datasets
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            return analysis
        else:
            return f"Error analyzing data: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def analyze_data_with_pandasai(df, question):
    """
    Use PandasAI to analyze the DataFrame based on a question
    
    Parameters:
        df: DataFrame to analyze (pandas or polars)
        question: string question to ask about the data
    """
    try:
        # Convert to pandas if using polars
        if st.session_state.use_polars and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        # Set up OpenAI with OpenRouter base URL
        llm = OpenAI(
            api_token=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-chat-v3-0324:free",
            custom_headers={"HTTP-Referer": "https://mcp-learning.example.com"}
        )
        
        # Get images dir for plots
        plots_dir = os.path.join(temp_data_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        # Initialize SmartDataframe with the LLM
        smart_df = SmartDataframe(df, config={"llm": llm, "save_charts": True, "save_charts_path": plots_dir})
            
        # Add analysis instructions to encourage visualizations
        if "plot" not in question.lower() and "graph" not in question.lower() and "chart" not in question.lower() and "visual" not in question.lower():
            # If the user hasn't explicitly asked for visualization, encourage it
            viz_prompt = f"{question}\n\nIf relevant, please include visualizations or plots to illustrate key patterns or insights in the data."
        else:
            viz_prompt = question
            
        # Execute the question
        with st.spinner("Analyzing with PandasAI..."):
            result = smart_df.chat(viz_prompt)
            
            # Check if any plots were generated
            plots = list_plots_in_output(str(result), plots_dir)
            
            # Return the result as a string, with plot display logic handled separately
            return {
                "text": str(result),
                "plots": plots
            }
    except Exception as e:
        import traceback
        return {
            "text": f"Error analyzing data with PandasAI: {str(e)}\n{traceback.format_exc()}",
            "plots": []
        }

def list_plots_in_output(result_text, plots_dir):
    """
    Scan the result text for references to generated plots
    and return a list of file paths to plots that actually exist
    """
    import re
    import os
    
    # Check if plots directory exists
    if not os.path.exists(plots_dir):
        return []
    
    # Get all image files in the plots directory
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
    
    # Sort by creation time (newest first)
    plot_files.sort(key=lambda x: os.path.getctime(os.path.join(plots_dir, x)), reverse=True)
    
    # Limit to 5 most recent files
    recent_plots = plot_files[:5]
    
    return [os.path.join(plots_dir, f) for f in recent_plots]

def analyze_data_hybrid(df, question, sample_rows=None, max_rows=1000):
    """
    Use both PandasAI and OpenRouter to analyze the DataFrame based on a question,
    then combine the results for a more comprehensive analysis
    
    Parameters:
        df: DataFrame to analyze (pandas or polars)
        question: string question to ask about the data
        sample_rows: if set, only use this many sample rows
        max_rows: maximum number of rows to send to OpenRouter
    """
    try:
        # Create placeholder for progress updates
        progress_container = st.empty()
        progress_container.info("Starting hybrid analysis...")
        
        # First try OpenRouter analysis as it's more reliable
        progress_container.info("Step 1/3: Analyzing with OpenRouter...")
        openrouter_result = analyze_data_with_openrouter(df, question, sample_rows, max_rows)
        
        # Now try PandasAI (which might fail on some Windows installations)
        plots = []
        try:
            progress_container.info("Step 2/3: Analyzing with PandasAI...")
            pandasai_result = analyze_data_with_pandasai(df, question)
            pandasai_text = pandasai_result["text"]
            plots = pandasai_result["plots"]
        except Exception as e:
            progress_container.warning("PandasAI analysis failed, continuing with OpenRouter only")
            pandasai_text = f"PandasAI analysis failed due to: {str(e)}"
        
        # Synthesize results
        progress_container.info("Step 3/3: Synthesizing insights...")
        
        # Create a unified prompt using both results
        combined_prompt = f"""I have analyzed a dataset using two different methods. 
Please combine these analyses into a comprehensive response:

ANALYSIS 1 (PandasAI):
{pandasai_text}

ANALYSIS 2 (OpenRouter direct):
{openrouter_result}

Please synthesize these insights into a coherent analysis that captures the best observations from both systems.
If one analysis failed or is incomplete, rely more heavily on the successful one.
"""
        
        # Initialize the LLM for synthesis
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://mcp-learning.example.com",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [
                {"role": "system", "content": "You are an expert at combining and synthesizing multiple data analyses into a coherent, insightful summary."},
                {"role": "user", "content": combined_prompt}
            ],
            "temperature": 0.2  # Lower temperature for more focused synthesis
        }
        
        # Make the API request for synthesis
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        # Clear the progress info
        progress_container.empty()
        
        if response.status_code == 200:
            result = response.json()
            combined_analysis = result['choices'][0]['message']['content']
            
            # Store original analyses for potential inspection
            if 'hybrid_original_analyses' not in st.session_state:
                st.session_state.hybrid_original_analyses = {}
                
            # Use timestamp to identify this analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.hybrid_original_analyses[timestamp] = {
                "pandasai": pandasai_text,
                "openrouter": openrouter_result,
                "combined": combined_analysis
            }
            
            # Store the timestamp for the latest analysis
            st.session_state.latest_hybrid_analysis = timestamp
            
            # Return combined analysis and plots
            return {
                "text": combined_analysis,
                "plots": plots
            }
        else:
            return {
                "text": f"Error synthesizing analyses: HTTP {response.status_code}. Individual results:\n\nPandasAI:\n{pandasai_text}\n\nOpenRouter:\n{openrouter_result}",
                "plots": plots
            }
    
    except Exception as e:
        import traceback
        return {
            "text": f"Error in hybrid analysis: {str(e)}\n{traceback.format_exc()}\n\nPartial results may be available.",
            "plots": []
        }

def analyze_data(df, question, sample_rows=None, max_rows=1000):
    """
    Analyze DataFrame using the selected backend
    """
    if st.session_state.analysis_backend == "pandasai":
        return analyze_data_with_pandasai(df, question)
    elif st.session_state.analysis_backend == "hybrid":
        return analyze_data_hybrid(df, question, sample_rows, max_rows)
    else:  # "openrouter"
        result = analyze_data_with_openrouter(df, question, sample_rows, max_rows)
        # Format to match the structure of other functions
        return {
            "text": result,
            "plots": []
        }

# Page config
st.set_page_config(
    page_title="MCP Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Title and server status
st.title("MCP Dashboard")
st.markdown("### Powered by OpenRouter & DeepSeek Chat v3 Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("GitHub Server Status")
    github_status = check_server_status(GITHUB_SERVER_URL)
    if github_status:
        st.success("✅ GitHub Server is running")
    else:
        st.error("❌ GitHub Server is not running. Ensure the GITHUB_SERVER_URL environment variable is set correctly.")

with col2:
    st.subheader("PostgreSQL Server Status")
    postgres_status = check_server_status(POSTGRES_SERVER_URL)
    if postgres_status:
        st.success("✅ PostgreSQL Server is running")
    else:
        st.error("❌ PostgreSQL Server is not running. Ensure the POSTGRES_SERVER_URL environment variable is set correctly.")

with col3:
    st.subheader("MySQL Server Status")
    mysql_status = check_server_status(MYSQL_SERVER_URL)
    if mysql_status:
        st.success("✅ MySQL Server is running")
    else:
        st.error("❌ MySQL Server is not running. Ensure the MYSQL_SERVER_URL environment variable is set correctly.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["GitHub Explorer", "PostgreSQL Explorer", "MySQL Explorer", "Data Analysis"])

with tab1:
    st.header("GitHub Explorer")
    
    if github_status:
        # GitHub Repository List
        if st.button("List My Repositories"):
            response = requests.get(f"{GITHUB_SERVER_URL}/repos")
            if response.status_code == 200:
                repos = response.json()
                if isinstance(repos, list):
                    df = pd.DataFrame(repos)
                    st.dataframe(df)
                else:
                    st.error(repos.get('error', 'Unknown error occurred'))
        
        # Repository Search
        st.subheader("Search Repositories")
        search_query = st.text_input("Enter search query")
        if search_query and st.button("Search"):
            response = requests.get(f"{GITHUB_SERVER_URL}/search", params={"query": search_query})
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list):
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                else:
                    st.error(results.get('error', 'Unknown error occurred'))
        
        # Repository Analysis with LLM
        st.subheader("Analyze Repository with LLM")
        repo_to_analyze = st.text_input("Enter repository name to analyze")
        if repo_to_analyze and st.button("Analyze Repository"):
            with st.spinner("Analyzing repository with AI..."):
                response = requests.get(f"{GITHUB_SERVER_URL}/analyze/{repo_to_analyze}")
                if response.status_code == 200:
                    result = response.json()
                    if 'error' in result and result['error']:
                        st.error(result['error'])
                    else:
                        st.write("### Repository Info")
                        st.json(result.get('repository_info', {}))
                        
                        st.write("### Key Files")
                        st.json(result.get('key_files', []))
                        
                        st.write("### AI Analysis")
                        analysis = result.get('analysis', 'No analysis available')
                        st.markdown(analysis)
        
        # Find Similar Repositories with LLM
        st.subheader("Find Similar Repositories")
        repo_to_find_similar = st.text_input("Enter repository name to find similar ones")
        if repo_to_find_similar and st.button("Find Similar"):
            with st.spinner("Finding similar repositories with AI..."):
                response = requests.get(f"{GITHUB_SERVER_URL}/find-similar/{repo_to_find_similar}")
                if response.status_code == 200:
                    result = response.json()
                    if 'error' in result and result['error']:
                        st.error(result['error'])
                    else:
                        st.write("### Repository Info")
                        st.write(f"Repository: {result.get('repository', '')}")
                        st.write(f"Language: {result.get('language', '')}")
                        st.write(f"Topics: {', '.join(result.get('topics', []))}")
                        
                        st.write("### Similar Repositories")
                        analysis = result.get('analysis', 'No analysis available')
                        st.markdown(analysis)
    else:
        st.warning("Please start the GitHub server first")

with tab2:
    st.header("PostgreSQL Explorer")
    
    if postgres_status:
        # List tables
        if st.button("List Tables"):
            response = requests.get(f"{POSTGRES_SERVER_URL}/tables")
            if response.status_code == 200:
                tables = response.json()
                if 'tables' in tables:
                    st.write("### Available Tables")
                    for table in tables['tables']:
                        st.write(f"- {table}")
                else:
                    st.error(tables.get('error', 'Unknown error occurred'))
        
        # Database query interface
        st.subheader("Query Database")
        query = st.text_area("Enter your SQL query")
        save_for_analysis = st.checkbox("Save results for analysis", value=True, key="save_sql_results")
        
        if query and st.button("Execute Query"):
            try:
                with st.spinner("Executing query..."):
                    st.info(f"Sending query to: {POSTGRES_SERVER_URL}/query")
                    st.code(query, language="sql")
                    
                    response = requests.post(
                        f"{POSTGRES_SERVER_URL}/query", 
                        json={"query": query},
                        timeout=60  # Increased timeout
                    )
                    
                    # Debug the response
                    st.info(f"Response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            
                            if isinstance(results, list):
                                if len(results) > 0:
                                    st.success(f"Query returned {len(results)} rows")
                                    df = pd.DataFrame(results)
                                    
                                    # Show column information
                                    st.write("### Column Information")
                                    col_info = pd.DataFrame({
                                        'Column': df.columns,
                                        'Type': df.dtypes.astype(str),
                                        'Non-Null Count': df.count().values,
                                        'First Value': [str(df[col].iloc[0]) if not df[col].isna().all() and len(df) > 0 else "NULL" for col in df.columns]
                                    })
                                    st.dataframe(col_info)
                                    
                                    # Store dataframe in session state
                                    if save_for_analysis:
                                        # Save to session state for later analysis
                                        st.session_state.last_query_df = df
                                        # Save to disk as a single temp file
                                        filepath = save_dataframe(df, "sql_query", reuse_current=False)
                                        st.session_state.last_query_filepath = filepath
                                        # Update the current temp file reference
                                        st.session_state.current_temp_file = filepath
                                        # Limit display size to prevent page freezing
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                        
                                        # Add option to download full result
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            "Download full results as CSV",
                                            csv,
                                            "query_results.csv",
                                            "text/csv",
                                            key='download-csv'
                                        )
                                    else:
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                else:
                                    st.info("Query executed successfully, but returned no results.")
                            elif 'error' in results:
                                st.error(results['error'])
                            else:
                                st.json(results)
                        except Exception as e:
                            st.error(f"Error processing results: {str(e)}")
                            st.text(f"Raw response: {response.text}")
                    else:
                        st.error(f"Error: Status code {response.status_code}")
                        try:
                            error_data = response.json()
                            st.error(json.dumps(error_data))
                        except:
                            st.error(response.text)
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Natural language query with OpenRouter
        st.subheader("Natural Language Query")
        st.markdown("Powered by DeepSeek Chat v3 model via OpenRouter")
        
        nl_query = st.text_input("Describe what you want to query")
        execute_query = st.checkbox("Execute the generated SQL", value=True, key="execute_nl_sql")
        save_nl_results = st.checkbox("Save results for analysis", value=True, key="save_nl_results")
        
        if nl_query and st.button("Generate & Execute SQL"):
            with st.spinner("Generating SQL with AI..."):
                try:
                    # Prepare the request
                    request_url = f"{POSTGRES_SERVER_URL}/nl_query"
                    request_payload = {"query": nl_query, "execute": str(execute_query).lower()}
                    
                    # Debug information
                    st.info(f"Sending request to: {request_url}")
                    st.info(f"Request payload: {json.dumps(request_payload)}")
                    
                    # Make the request with increased timeout
                    response = requests.post(
                        request_url,
                        json=request_payload,
                        timeout=60  # Increased timeout for slow responses
                    )
                    
                    # Debug the response
                    st.info(f"Response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        # Try to convert response to JSON
                        try:
                            results = response.json()
                            st.info(f"Parsed results: {json.dumps(results)}")
                            
                            if 'sql' in results:
                                st.write("### Generated SQL")
                                st.code(results['sql'], language='sql')
                                
                                if 'explanation' in results and results['explanation']:
                                    st.write("### Explanation")
                                    st.markdown(results['explanation'])
                                
                                if execute_query and 'results' in results:
                                    st.write("### Query Results")
                                    if isinstance(results['results'], list):
                                        if len(results['results']) > 0:
                                            df = pd.DataFrame(results['results'])
                                            
                                            # Store dataframe in session state
                                            if save_nl_results:
                                                # Save to session state for later analysis
                                                st.session_state.last_query_df = df
                                                # Save to disk as a single temp file
                                                filepath = save_dataframe(df, "nl_query", reuse_current=False)
                                                st.session_state.last_query_filepath = filepath
                                                # Update the current temp file reference
                                                st.session_state.current_temp_file = filepath
                                                # Limit display size to prevent page freezing
                                                row_count_display = len(df)
                                                st.dataframe(df.head(5))
                                                st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                                
                                                # Add option to download full result
                                                csv = df.to_csv(index=False).encode('utf-8')
                                                st.download_button(
                                                    "Download full results as CSV",
                                                    csv,
                                                    "query_results.csv",
                                                    "text/csv",
                                                    key='download-csv'
                                                )
                                            else:
                                                row_count_display = len(df)
                                                st.dataframe(df.head(5))
                                                st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                        else:
                                            st.info("Query executed successfully, but returned no results.")
                                    else:
                                        st.warning("Results are not in the expected list format.")
                                        st.json(results['results'])
                                elif execute_query and 'error' in results and results['error']:
                                    st.error(f"Error executing SQL: {results['error']}")
                                elif execute_query:
                                    st.warning("The query was not executed or no results were returned.")
                                    # Add a manual execute button
                                    if st.button("Execute this SQL manually"):
                                        with st.spinner("Executing SQL..."):
                                            sql_query = results['sql']
                                            sql_response = requests.post(
                                                f"{POSTGRES_SERVER_URL}/query", 
                                                json={"query": sql_query},
                                                timeout=60
                                            )
                                            if sql_response.status_code == 200:
                                                sql_results = sql_response.json()
                                                if isinstance(sql_results, list) and len(sql_results) > 0:
                                                    sql_df = pd.DataFrame(sql_results)
                                                    st.dataframe(sql_df.head(100))
                                                    st.success(f"Retrieved {len(sql_results)} rows.")
                                                else:
                                                    st.info("Query executed but returned no results.")
                                            else:
                                                st.error(f"Error: {sql_response.status_code} - {sql_response.text}")
                            else:
                                st.error("No SQL was generated. Response may be malformed.")
                                st.json(results)
                        except Exception as e:
                            st.error(f"Error parsing response JSON: {str(e)}")
                            st.text(f"Raw response: {response.text}")
                    else:
                        st.error(f"Server returned status code {response.status_code}")
                        try:
                            error_details = response.json()
                            st.error(f"Error details: {json.dumps(error_details)}")
                        except:
                            st.error(f"Raw response: {response.text}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Get table schema
        st.subheader("View Table Schema")
        schema_table_name = st.text_input("Enter table name")
        if schema_table_name and st.button("View Schema"):
            response = requests.get(f"{POSTGRES_SERVER_URL}/schema/{schema_table_name}")
            if response.status_code == 200:
                schema = response.json()
                st.write(f"### Schema for {schema_table_name}")
                if 'columns' in schema:
                    columns_df = pd.DataFrame(schema['columns'])
                    st.dataframe(columns_df)
                else:
                    st.error(schema.get('detail', 'Unknown error occurred'))
    else:
        st.warning("Please start the PostgreSQL server first")

with tab3:
    st.header("MySQL Explorer")
    
    if mysql_status:
        # List tables
        if st.button("List Tables", key="mysql_list_tables"):
            response = requests.get(f"{MYSQL_SERVER_URL}/tables")
            if response.status_code == 200:
                tables = response.json()
                if 'tables' in tables:
                    st.write("### Available Tables")
                    for table in tables['tables']:
                        st.write(f"- {table}")
                else:
                    st.error(tables.get('error', 'Unknown error occurred'))
        
        # Database query interface
        st.subheader("Query Database")
        mysql_query = st.text_area("Enter your MySQL query", key="mysql_query_area")
        save_mysql_for_analysis = st.checkbox("Save results for analysis", value=True, key="save_mysql_results")
        
        if mysql_query and st.button("Execute MySQL Query", key="execute_mysql"):
            try:
                with st.spinner("Executing MySQL query..."):
                    st.info(f"Sending query to: {MYSQL_SERVER_URL}/query")
                    st.code(mysql_query, language="sql")
                    
                    response = requests.post(
                        f"{MYSQL_SERVER_URL}/query", 
                        json={"query": mysql_query},
                        timeout=60  # Increased timeout
                    )
                    
                    # Debug the response
                    st.info(f"Response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            
                            if 'results' in results and isinstance(results['results'], list):
                                result_rows = results['results']
                                if len(result_rows) > 0:
                                    st.success(f"Query returned {len(result_rows)} rows")
                                    df = pd.DataFrame(result_rows)
                                    
                                    # Show column information
                                    st.write("### Column Information")
                                    col_info = pd.DataFrame({
                                        'Column': df.columns,
                                        'Type': df.dtypes.astype(str),
                                        'Non-Null Count': df.count().values,
                                        'First Value': [str(df[col].iloc[0]) if not df[col].isna().all() and len(df) > 0 else "NULL" for col in df.columns]
                                    })
                                    st.dataframe(col_info)
                                    
                                    # Store dataframe in session state
                                    if save_mysql_for_analysis:
                                        # Save to session state for later analysis
                                        st.session_state.last_query_df = df
                                        # Save to disk as a single temp file
                                        filepath = save_dataframe(df, "mysql_query", reuse_current=False)
                                        st.session_state.last_query_filepath = filepath
                                        # Update the current temp file reference
                                        st.session_state.current_temp_file = filepath
                                        # Limit display size to prevent page freezing
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                        
                                        # Add option to download full result
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            "Download full results as CSV",
                                            csv,
                                            "mysql_query_results.csv",
                                            "text/csv",
                                            key='download-mysql-csv'
                                        )
                                    else:
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                else:
                                    st.info("Query executed successfully, but returned no results.")
                            elif 'error' in results:
                                st.error(results['error'])
                            else:
                                st.json(results)
                        except Exception as e:
                            st.error(f"Error processing results: {str(e)}")
                            st.text(f"Raw response: {response.text}")
                    else:
                        st.error(f"Error: Status code {response.status_code}")
                        try:
                            error_data = response.json()
                            st.error(json.dumps(error_data))
                        except:
                            st.error(response.text)
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Natural language query with OpenRouter
        st.subheader("Natural Language Query")
        st.markdown("Powered by DeepSeek Chat v3 model via OpenRouter")
        
        mysql_nl_query = st.text_input("Describe what you want to query", key="mysql_nl_query")
        execute_mysql_query = st.checkbox("Execute the generated SQL", value=True, key="execute_mysql_nl_sql")
        save_mysql_nl_results = st.checkbox("Save results for analysis", value=True, key="save_mysql_nl_results")
        
        if mysql_nl_query and st.button("Generate & Execute MySQL SQL", key="generate_mysql_sql"):
            with st.spinner("Generating MySQL SQL with AI..."):
                try:
                    # Specify db_type="mysql" for the MySQL dialect
                    response = requests.post(
                        f"{MYSQL_SERVER_URL}/nl_query", 
                        json={
                            "text": mysql_nl_query,
                            "execute": str(execute_mysql_query).lower()
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.write("### Generated SQL")
                        st.code(result['sql'], language="sql")
                        
                        if execute_mysql_query:
                            if 'results' in result and isinstance(result['results'], list):
                                result_rows = result['results']
                                if len(result_rows) > 0:
                                    st.success(f"Query returned {len(result_rows)} rows")
                                    df = pd.DataFrame(result_rows)
                                    
                                    # Store results for analysis if requested
                                    if save_mysql_nl_results:
                                        # Save to session state for later analysis
                                        st.session_state.last_query_df = df
                                        # Save to disk
                                        filepath = save_dataframe(df, "mysql_nl_query", reuse_current=False)
                                        st.session_state.last_query_filepath = filepath
                                        # Update the current temp file reference
                                        st.session_state.current_temp_file = filepath
                                        # Limit display size to prevent page freezing
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                        
                                        # Add option to download full result
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            "Download full results as CSV",
                                            csv,
                                            "mysql_nl_query_results.csv",
                                            "text/csv",
                                            key='download-mysql-nl-csv'
                                        )
                                    else:
                                        row_count_display = len(df)
                                        st.dataframe(df.head(5))
                                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                                else:
                                    st.info("Query executed successfully, but returned no results.")
                                    if 'debug_info' in result:
                                        st.info(f"Debug information: {result['debug_info']}")
                            elif 'error' in result and result['error']:
                                st.error(f"Error executing query: {result['error']}")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error in natural language query: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    else:
        st.warning("Please start the MySQL server first")

with tab4:
    st.header("Data Analysis")
    st.markdown("Analyze your query results with AI")
    
    # Add a sidebar for settings
    with st.sidebar:
        st.header("Analysis Settings")
        
        # Toggle for Polars vs Pandas
        st.session_state.use_polars = st.toggle(
            "Use Polars (faster)",
            value=st.session_state.use_polars,
            help="Toggle between Polars (faster) and Pandas (more compatible)"
        )
        
        # Backend selection for LLM analysis
        analysis_backend = st.radio(
            "Analysis Backend:",
            ["OpenRouter", "PandasAI", "Hybrid"],
            index=0 if st.session_state.analysis_backend == "openrouter" else 1 if st.session_state.analysis_backend == "pandasai" else 2,
            help="Select which backend to use for data analysis"
        )
        
        if analysis_backend == "OpenRouter":
            st.session_state.analysis_backend = "openrouter"
        elif analysis_backend == "PandasAI":
            st.session_state.analysis_backend = "pandasai"
        else:
            st.session_state.analysis_backend = "hybrid"
            
        # Show current backend information
        if st.session_state.analysis_backend == "openrouter":
            st.info("Using OpenRouter with DeepSeek Chat v3 model - most reliable option")
        elif st.session_state.analysis_backend == "pandasai":
            st.info("Using PandasAI v3 with OpenRouter LLM - best for data visualizations and statistical insights")
        else:
            st.info("Using Hybrid analysis: Combines both methods for comprehensive insights, with fallback if PandasAI has compatibility issues")
            
        # Additional explanation for each backend
        with st.expander("About Analysis Backends"):
            st.markdown("""
            - **OpenRouter**: Direct analysis using DeepSeek Chat v3 with structured data views. Good for general insights and patterns. Most reliable option.
            - **PandasAI**: Uses PandasAI v3's specialized data analysis capabilities with auto-generated visualizations. Best for statistical insights and data transformations with visual aids.
            - **Hybrid**: Attempts to use both approaches and synthesize results for the most comprehensive analysis.
            """)
            
            # Add link to PandasAI repository
            st.markdown("""
            [Learn more about PandasAI on GitHub →](https://github.com/sinaptik-ai/pandas-ai)
            """)
            
            # Display that we're using PandasAI v3
            st.caption("This system uses PandasAI v3, a complete redesign with improved visualization capabilities.")
        
        # Option to analyze full dataset or just a sample (only relevant for OpenRouter and Hybrid)
        if st.session_state.analysis_backend in ["openrouter", "hybrid"]:
            use_full_dataset = st.checkbox("Analyze full dataset (may be slower but more accurate)", value=True)
        else:
            use_full_dataset = True  # PandasAI always uses the full dataset
        
        st.markdown("---")
        st.markdown(
            "**Using:** " + 
            ("**Polars** (faster for large datasets)" if st.session_state.use_polars else 
             "**Pandas** (more compatibility)")
        )
        
        # Display current temp file info if it exists
        if st.session_state.current_temp_file and os.path.exists(st.session_state.current_temp_file):
            file_size = os.path.getsize(st.session_state.current_temp_file) / (1024 * 1024)  # Convert to MB
            st.markdown(f"**Current temp file:** {os.path.basename(st.session_state.current_temp_file)}")
            st.markdown(f"**Size:** {file_size:.2f} MB")
    
    # Check if there's a dataframe in session state
    if hasattr(st.session_state, 'last_query_df') and isinstance(st.session_state.last_query_df, (pd.DataFrame, pl.DataFrame)):
        df = st.session_state.last_query_df
        
        # Convert if needed
        if st.session_state.use_polars and isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
            st.session_state.last_query_df = df
        elif not st.session_state.use_polars and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            st.session_state.last_query_df = df
        
        # Display info and sample rows
        if st.session_state.use_polars:
            st.write(f"### Available Data ({len(df)} rows, {len(df.columns)} columns)")
            st.dataframe(df.head(5).to_pandas())  # Convert to pandas for display
        else:
            st.write(f"### Available Data ({len(df)} rows, {len(df.columns)} columns)")
            st.dataframe(df.head(5))
        
        st.subheader("Ask Questions About Your Data")
        
        # New Analysis button to clear previous context
        if hasattr(st.session_state, 'show_followup') and st.session_state.show_followup:
            if st.button("Start New Analysis", key="new_analysis"):
                # Clear previous analysis data but keep the dataframe
                if hasattr(st.session_state, 'last_analysis'):
                    del st.session_state.last_analysis
                if hasattr(st.session_state, 'show_followup'):
                    del st.session_state.show_followup
                # Reset temp file tracking
                st.session_state.current_temp_file = None
                # Clean up temp files
                clean_temp_files(keep_current=False)
                st.rerun()
        
        data_question = st.text_area("What would you like to know about this data?", 
                                    placeholder="e.g., What are the main patterns in this data? or Summarize the key statistics.")
        
        if data_question and st.button("Analyze Data"):
            with st.spinner("Analyzing data with AI..."):
                # Use None for sample_rows if full dataset is selected, otherwise use 5 rows
                sample_size = None if use_full_dataset else 5
                analysis = analyze_data(df, data_question, sample_rows=sample_size)
                st.markdown("### Analysis")
                st.markdown(analysis["text"])
                
                # Display plots if any were generated
                if analysis["plots"] and len(analysis["plots"]) > 0:
                    st.markdown("### Visualizations")
                    plot_cols = st.columns(min(3, len(analysis["plots"])))  # Up to 3 columns
                    
                    for i, plot_path in enumerate(analysis["plots"]):
                        try:
                            col_idx = i % len(plot_cols)
                            with plot_cols[col_idx]:
                                # Read the image and display it
                                img = Image.open(plot_path)
                                st.image(img, caption=f"Plot {i+1}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error displaying plot {i+1}: {str(e)}")
                
                # For PandasAI, also save the results to ensure they can be used in follow-up
                if st.session_state.current_temp_file is None:
                    filepath = save_dataframe(df, "analysis", reuse_current=False)
                
                # Option to follow up
                st.session_state.last_analysis = analysis["text"]
                # Save plots in session state for follow-up
                st.session_state.last_analysis_plots = analysis["plots"]
                st.session_state.show_followup = True
                
                # For hybrid mode, give option to view individual analyses
                if st.session_state.analysis_backend == "hybrid" and hasattr(st.session_state, 'latest_hybrid_analysis'):
                    timestamp = st.session_state.latest_hybrid_analysis
                    if timestamp in st.session_state.hybrid_original_analyses:
                        with st.expander("View Individual Analyses"):
                            individual_analyses = st.session_state.hybrid_original_analyses[timestamp]
                            
                            st.markdown("### PandasAI Analysis")
                            st.markdown(individual_analyses["pandasai"])
                            
                            st.markdown("### OpenRouter Analysis")
                            st.markdown(individual_analyses["openrouter"])
        
        # Show follow-up area if there was a previous analysis
        if hasattr(st.session_state, 'show_followup') and st.session_state.show_followup:
            st.subheader("Follow-up Question")
            followup_question = st.text_area("Ask a follow-up question based on the analysis:", 
                                           placeholder="e.g., Can you provide more details about...?")
            
            if followup_question and st.button("Get Follow-up Analysis"):
                with st.spinner("Analyzing further..."):
                    # Include previous analysis in the context
                    combined_question = f"Previous analysis: {st.session_state.last_analysis}\n\nFollow-up question: {followup_question}"
                    # Use same sample size setting as initial analysis
                    sample_size = None if use_full_dataset else 5
                    
                    # For follow-up, reuse the existing temp file
                    if st.session_state.current_temp_file:
                        save_dataframe(df, "analysis", reuse_current=True)
                        
                    followup_analysis = analyze_data(df, combined_question, sample_rows=sample_size)
                    st.markdown("### Follow-up Analysis")
                    st.markdown(followup_analysis["text"])
                    
                    # Display plots for follow-up analysis
                    if followup_analysis["plots"] and len(followup_analysis["plots"]) > 0:
                        st.markdown("### Follow-up Visualizations")
                        plot_cols = st.columns(min(3, len(followup_analysis["plots"])))  # Up to 3 columns
                        
                        for i, plot_path in enumerate(followup_analysis["plots"]):
                            try:
                                col_idx = i % len(plot_cols)
                                with plot_cols[col_idx]:
                                    # Read the image and display it
                                    img = Image.open(plot_path)
                                    st.image(img, caption=f"Plot {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error displaying plot {i+1}: {str(e)}")
                    
                    # For hybrid mode, give option to view individual analyses for follow-up
                    if st.session_state.analysis_backend == "hybrid" and hasattr(st.session_state, 'latest_hybrid_analysis'):
                        timestamp = st.session_state.latest_hybrid_analysis
                        if timestamp in st.session_state.hybrid_original_analyses:
                            with st.expander("View Individual Analyses for Follow-up"):
                                individual_analyses = st.session_state.hybrid_original_analyses[timestamp]
                                
                                st.markdown("### PandasAI Analysis")
                                st.markdown(individual_analyses["pandasai"])
                                
                                st.markdown("### OpenRouter Analysis")
                                st.markdown(individual_analyses["openrouter"])
        
        # Option to clear the current data
        if st.button("Clear Current Data"):
            if hasattr(st.session_state, 'last_query_df'):
                del st.session_state.last_query_df
            if hasattr(st.session_state, 'last_query_filepath'):
                del st.session_state.last_query_filepath
            if hasattr(st.session_state, 'last_analysis'):
                del st.session_state.last_analysis
            if hasattr(st.session_state, 'show_followup'):
                del st.session_state.show_followup
            # Clear the temp file reference
            st.session_state.current_temp_file = None
            # Clean up temp files
            clean_temp_files(keep_current=False)
            st.rerun()
    else:
        st.info("No data available for analysis. Run a query in the MySQL Explorer tab and save the results for analysis.")
        
        # Add option to use PandasAI's chat with multiple dataframes
        st.subheader("Multi-DataFrame Analysis")
        st.markdown("Load multiple CSV or Parquet files for combined analysis")
        
        # List saved data files for multi-dataframe loading
        if os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
            # Get both CSV and Parquet files
            data_files = [f for f in os.listdir(temp_data_dir) if f.endswith(('.csv', '.parquet'))]
            
            if data_files:
                # Multi-select for files
                selected_files = st.multiselect(
                    "Select files to analyze together:",
                    data_files,
                    help="You can select multiple files to analyze relationships between them"
                )
                
                if selected_files and len(selected_files) > 1:
                    if st.button("Load Selected Files for Multi-Analysis"):
                        try:
                            # Load each file into a dataframe
                            dfs = []
                            for file in selected_files:
                                file_path = os.path.join(temp_data_dir, file)
                                if file.endswith('.csv'):
                                    if st.session_state.use_polars:
                                        df = pl.read_csv(file_path)
                                    else:
                                        df = pd.read_csv(file_path)
                                else:  # parquet
                                    if st.session_state.use_polars:
                                        df = pl.read_parquet(file_path)
                                    else:
                                        df = pd.read_parquet(file_path)
                                
                                # Add filename as identifier
                                file_id = os.path.splitext(file)[0]  # Remove extension
                                if st.session_state.use_polars:
                                    # Store as tuple with identifier
                                    dfs.append((file_id, df))
                                else:
                                    dfs.append((file_id, df))
                            
                            # Store in session state
                            st.session_state.multi_dfs = dfs
                            st.success(f"Loaded {len(dfs)} dataframes for multi-analysis")
                            
                            # Show quick info about loaded dataframes
                            st.subheader("Loaded DataFrames")
                            for name, df in dfs:
                                st.write(f"**{name}**: {len(df)} rows, {len(df.columns) if hasattr(df, 'columns') else 0} columns")
                            
                            # Form for asking questions about multiple dataframes
                            st.subheader("Ask Questions About Multiple DataFrames")
                            multi_question = st.text_area(
                                "What would you like to know about these datasets?",
                                placeholder="e.g., What's the relationship between these datasets? or Compare the key metrics across these tables."
                            )
                            
                            if multi_question and st.button("Analyze Multiple DataFrames"):
                                with st.spinner("Analyzing relationships between dataframes..."):
                                    # Set up OpenAI with OpenRouter base URL
                                    llm = OpenAI(
                                        api_token=os.getenv('OPENROUTER_API_KEY'),
                                        base_url="https://openrouter.ai/api/v1",
                                        model="deepseek/deepseek-chat-v3-0324:free",
                                        custom_headers={"HTTP-Referer": "https://mcp-learning.example.com"}
                                    )
                                    
                                    # Convert each dataframe to SmartDataframe
                                    smart_dfs = []
                                    for name, df in dfs:
                                        if st.session_state.use_polars:
                                            df = df.to_pandas()
                                        smart_dfs.append(SmartDataframe(df, name=name, config={"llm": llm}))
                                    
                                    # Create a multi-df prompt (we'll use a single SmartDataframe with multi-df context)
                                    # Use the first dataframe as the primary one
                                    primary_df = smart_dfs[0]
                                    context = f"I have {len(smart_dfs)} related dataframes with the following names and structures:\n\n"
                                    
                                    for i, smart_df in enumerate(smart_dfs):
                                        df_name = smart_df._name
                                        df = smart_df._df
                                        context += f"DataFrame {i+1}: {df_name}\n"
                                        context += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                                        context += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                                    
                                    # Add the multi-df context to the question
                                    enhanced_question = f"{context}\n\nQuestion: {multi_question}"
                                    
                                    # Run the analysis on the primary dataframe with context about all dataframes
                                    result = primary_df.chat(enhanced_question)
                                    
                                    st.markdown("### Analysis Results")
                                    st.markdown(str(result))
                        except Exception as e:
                            st.error(f"Error loading files for multi-analysis: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        
        # List saved data files (single file loading)
        st.subheader("Previously Saved Data Files")
        
        if os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
            # Get both CSV and Parquet files
            csv_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.csv')]
            parquet_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.parquet')]
            
            if csv_files or parquet_files:
                file_type = st.radio("File type:", ["CSV", "Parquet"], 
                                    index=1 if st.session_state.use_polars else 0,
                                    help="CSV files are created with pandas, Parquet files with Polars")
                
                if file_type == "CSV" and csv_files:
                    selected_file = st.selectbox("Select a saved CSV file to load:", csv_files)
                    if selected_file and st.button("Load Selected File"):
                        try:
                            file_path = os.path.join(temp_data_dir, selected_file)
                            if st.session_state.use_polars:
                                df = pl.read_csv(file_path)
                            else:
                                df = pd.read_csv(file_path)
                                
                            st.session_state.last_query_df = df
                            st.session_state.last_query_filepath = file_path
                            st.success(f"Loaded data with {len(df)} rows.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
                
                elif file_type == "Parquet" and parquet_files:
                    selected_file = st.selectbox("Select a saved Parquet file to load:", parquet_files)
                    if selected_file and st.button("Load Selected File"):
                        try:
                            file_path = os.path.join(temp_data_dir, selected_file)
                            if st.session_state.use_polars:
                                df = pl.read_parquet(file_path)
                            else:
                                # For pandas we need to use pyarrow
                                df = pd.read_parquet(file_path)
                                
                            st.session_state.last_query_df = df
                            st.session_state.last_query_filepath = file_path
                            st.success(f"Loaded data with {len(df)} rows.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
                else:
                    st.info(f"No {file_type} files found in the temp directory.")
            else:
                st.info("No saved data files found.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, FastAPI, PyGithub, and OpenRouter")

# Cleanup when the app is closed
def cleanup():
    stop_server(github_server_process)
    stop_server(postgres_server_process)
    stop_server(mysql_server_process)
    clean_temp_files(keep_current=False)  # Clean all temp files on app exit

# Register cleanup function
st.session_state.cleanup = cleanup

def clean_temp_files(keep_current=True):
    """
    Clean up old temp files from the temp directory
    
    Parameters:
        keep_current: If True, keeps the current temp file (st.session_state.current_temp_file)
    """
    if not os.path.exists(temp_data_dir):
        return
        
    # Get list of all temp files
    all_files = [os.path.join(temp_data_dir, f) for f in os.listdir(temp_data_dir) 
                if f.endswith('.csv') or f.endswith('.parquet')]
    
    # Keep the current temp file if requested
    current_temp_file = st.session_state.current_temp_file
    
    for file_path in all_files:
        # Skip the current temp file if keep_current is True
        if keep_current and file_path == current_temp_file:
            continue
            
        try:
            os.remove(file_path)
        except Exception as e:
            # Silently fail if we can't delete a file
            pass
            