import os
from typing import Dict, List, Optional, Any
import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import logging
import argparse
from dotenv import load_dotenv
from openrouter_utils import nl_to_sql
import re

# Parse command line arguments
parser = argparse.ArgumentParser(description='MySQL API Server')
parser.add_argument('--port', type=int, default=8006, help='Port to run the server on')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MySQL Database Server",
    description="API for interacting with MySQL databases",
    version="1.0.0"
)

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_NAME', os.getenv('MYSQL_DATABASE')),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'ssl_ca': os.getenv('MYSQL_SSL_CA', '')
}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests to the server"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

def get_db_connection():
    """Create and return a database connection"""
    try:
        # Configure SSL if CA certificate is provided
        ssl_args = {}
        if DB_CONFIG['ssl_ca']:
            ssl_args = {'ssl_ca': DB_CONFIG['ssl_ca']}
        
        # Remove ssl_ca from connection params as it's not a valid parameter
        connection_params = {k: v for k, v in DB_CONFIG.items() if k != 'ssl_ca'}
        
        # Add SSL arguments if needed
        if ssl_args:
            connection_params['ssl_disabled'] = False
            connection_params.update(ssl_args)
        
        conn = mysql.connector.connect(**connection_params)
        return conn
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def get_schema_info() -> str:
    """Get database schema information for better LLM context"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s
        """, (DB_CONFIG['database'],))
        tables = [table[0] for table in cursor.fetchall()]
        
        # For each table, get column information
        schema_info = "Available tables (exact case-sensitive names):\n"
        for table in tables:
            schema_info += f"- `{table}`\n"
        
        schema_info += "\nDetailed schema information:\n"
        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """, (DB_CONFIG['database'], table))
            columns = cursor.fetchall()
            
            schema_info += f"Table: `{table}`\n"
            schema_info += "Columns:\n"
            for col in columns:
                schema_info += f"  - `{col[0]}` ({col[1]}, nullable: {col[2]})\n"
            schema_info += "\n"
        
        return schema_info
    except Exception as e:
        logger.error(f"Error getting schema info: {e}")
        return ""
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/")
async def root():
    """Root endpoint to check server status"""
    return {
        "status": "running",
        "message": "MySQL Database Server is running",
        "endpoints": {
            "/tables": "List all tables",
            "/schema/{table_name}": "Get table schema",
            "/query": "Execute SQL query",
            "/nl_query": "Convert natural language to SQL using OpenRouter",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "openrouter": "configured" if os.getenv('OPENROUTER_API_KEY') else "not_configured"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": f"error: {str(e)}",
            "openrouter": "configured" if os.getenv('OPENROUTER_API_KEY') else "not_configured"
        }

@app.get("/tables")
async def list_tables():
    """List all tables in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s
        """, (DB_CONFIG['database'],))
        tables = cursor.fetchall()
        return {"tables": [table[0] for table in tables]}
    except Error as e:
        logger.error(f"Error listing tables: {e}")
        return {"error": str(e), "tables": []}
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    """Get the schema for a specific table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """, (DB_CONFIG['database'], table_name))
        columns = cursor.fetchall()
        if not columns:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        schema = {
            "table": table_name,
            "columns": [{
                "name": col[0],
                "type": col[1],
                "nullable": col[2]
            } for col in columns]
        }
        return schema
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/query")
async def execute_query(query: Dict[str, str]):
    """Execute a read-only SQL query"""
    if 'query' not in query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    # Basic query validation
    sql = query['query'].strip().lower()
    if any(cmd in sql for cmd in ['insert', 'update', 'delete', 'drop', 'alter', 'create']):
        raise HTTPException(status_code=400, detail="Only read-only queries are allowed")
    
    # Extract the limit from the query to check if it's reasonable
    limit_match = re.search(r'limit\s+(\d+)', sql)
    limit = int(limit_match.group(1)) if limit_match else None
    
    # Apply a reasonable default limit if none is specified
    # or if the specified limit is too high
    if not limit or limit > 1000:
        # If there's no limit specified, add one to prevent large result sets
        if not limit:
            if 'limit' not in sql:
                sql += ' LIMIT 1000'
            # If limit is too high, warn about it
        else:
            logger.warning(f"Requested limit of {limit} rows is too high. Capping at 1000.")
            sql = re.sub(r'limit\s+\d+', 'LIMIT 1000', sql)
    
    # Execute the query
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # For large results, provide info about row count
        result_count = len(results)
        
        # Return query results
        return {
            "results": results,
            "row_count": result_count,
            "query": sql
        }
    except Error as e:
        # Return error details
        logger.error(f"Error executing query: {e}")
        return {
            "error": str(e),
            "query": sql
        }
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/nl_query")
async def natural_language_query(query: Dict[str, Any]):
    """
    Convert natural language to SQL using OpenRouter, then execute the query
    
    This function takes a natural language query about the database,
    uses an LLM to convert it to SQL, and then executes the query.
    
    Parameters:
    - query: Dict with keys:
      - "text": Natural language query text
      - "execute": Boolean flag to indicate if the query should be executed
    
    Returns:
    - Dict with keys:
      - "nl_query": Original natural language query
      - "sql": Generated SQL query
      - "results": Query results (if executed)
      - "error": Error message (if any)
    """
    # Validate input
    if 'text' not in query:
        raise HTTPException(status_code=400, detail="No query text provided")
    
    nl_query = query['text']
    execute = query.get('execute', 'true')  # Default to executing the query
    # Convert to boolean if it comes as string
    if isinstance(execute, str):
        execute = execute.lower() == 'true'
    
    # Initialize result
    result = {
        "nl_query": nl_query,
        "sql": None,
        "results": [],
        "row_count": 0,
        "error": None
    }
    
    # Get database schema to provide context for the LLM
    schema_info = get_schema_info()
    
    # Convert natural language to SQL
    sql_response = nl_to_sql(nl_query, schema_info, db_type="mysql")
    
    # Handle different response formats (could be string or dict)
    if isinstance(sql_response, dict) and 'sql' in sql_response:
        sql = sql_response['sql']
        result['sql'] = sql
    else:
        sql = sql_response
        result['sql'] = sql
    
    # Execute the query if requested
    if sql and execute:
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            try:
                logger.info(f"Executing SQL query: {sql}")
                cursor.execute(sql)
                results = cursor.fetchall()
                result['results'] = results
                result['row_count'] = len(results)
                
                if len(results) == 0:
                    logger.warning(f"Query returned 0 rows: {sql}")
                    # Try checking if the table exists
                    table_name = ""
                    if "FROM" in sql.upper():
                        # Extract table name from query
                        table_match = re.search(r'FROM\s+`?([^`\s]+)`?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            # Check if table exists
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'")
                                table_exists = cursor.fetchone()
                                result['debug_info'] = f"Table '{table_name}' exists: {table_exists[0] > 0}"
                                
                                if table_exists[0] > 0:
                                    # Table exists, check row count
                                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                                    row_count = cursor.fetchone()
                                    result['debug_info'] += f". Table has {row_count[0]} rows."
                            except Exception as table_check_error:
                                logger.error(f"Error checking table: {str(table_check_error)}")
                                result['debug_info'] = f"Error checking table '{table_name}': {str(table_check_error)}"
            except Error as e:
                error_msg = f"Error executing SQL: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
        except Exception as e:
            error_msg = f"Error connecting to database: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
        finally:
            if 'conn' in locals():
                conn.close()
    else:
        logger.warning(f"Query not executed. Execute flag: {execute}, SQL present: {'sql' in result}")
    
    return result

if __name__ == "__main__":
    PORT = int(os.getenv('MYSQL_SERVER_PORT', args.port))
    logger.info(f"Starting MySQL server on http://localhost:{PORT}")
    logger.info(f"Database config: {DB_CONFIG['host']}:{DB_CONFIG['port']} - {DB_CONFIG['database']}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 