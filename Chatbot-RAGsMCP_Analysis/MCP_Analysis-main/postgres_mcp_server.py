import os
from typing import Dict, List, Optional
import psycopg2
from psycopg2 import Error
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import logging
import argparse
from dotenv import load_dotenv
from openrouter_utils import nl_to_sql
import re

# Parse command line arguments
parser = argparse.ArgumentParser(description='PostgreSQL API Server')
parser.add_argument('--port', type=int, default=8004, help='Port to run the server on')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PostgreSQL Database Server",
    description="API for interacting with PostgreSQL databases",
    version="1.0.0"
)

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', os.getenv('MYSQL_HOST')),
    'user': os.getenv('POSTGRES_USER', os.getenv('MYSQL_USER')),
    'password': os.getenv('POSTGRES_PASSWORD', os.getenv('MYSQL_PASSWORD')),
    'database': os.getenv('POSTGRES_DATABASE', os.getenv('MYSQL_DATABASE')),
    'port': int(os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432')))
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
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Error as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
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
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        
        # For each table, get column information
        schema_info = "Available tables (exact case-sensitive names):\n"
        for table in tables:
            schema_info += f"- \"{table}\"\n"
        
        schema_info += "\nDetailed schema information:\n"
        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            columns = cursor.fetchall()
            
            schema_info += f"Table: \"{table}\"\n"
            schema_info += "Columns:\n"
            for col in columns:
                schema_info += f"  - \"{col[0]}\" ({col[1]})\n"
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
        "message": "PostgreSQL Database Server is running",
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
            WHERE table_schema = 'public'
        """)
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
            WHERE table_schema = 'public' AND table_name = %s
        """, (table_name,))
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
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > 1000:  # If limit is too high, enforce a lower limit
            logger.warning(f"Query had excessive LIMIT {limit_value}, reducing to 1000")
            sql = re.sub(r'limit\s+\d+', 'LIMIT 1000', sql)
    elif not 'limit' in sql.lower():  # If no limit specified, add a default limit
        if sql.strip().endswith(';'):
            sql = sql[:-1] + " LIMIT 1000;"
        else:
            sql = sql + " LIMIT 1000;"
    
    try:
        conn = get_db_connection()
        # Set a statement timeout (5 seconds)
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout TO 5000;")  # 5 seconds
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/nl_query")
async def natural_language_query(query: Dict[str, str]):
    """Convert natural language to SQL using OpenRouter and execute"""
    logger.info(f"Received natural language query: {query}")
    
    if 'query' not in query:
        logger.error("No query provided in request")
        return {"error": "No query provided", "sql": "", "explanation": ""}
    
    nl_query = query['query']
    
    # Get schema information for better context
    schema_info = get_schema_info()
    logger.info(f"Schema info: {schema_info[:100]}...")  # Log first 100 chars
    
    # Convert natural language to SQL using OpenRouter
    result = nl_to_sql(nl_query, schema_info)
    logger.info(f"NL-to-SQL result: {result}")
    
    # Check if we want to execute the query as well
    execute = query.get('execute', 'false')
    # Convert to boolean if it comes as string
    if isinstance(execute, str):
        execute = execute.lower() == 'true'
    
    if execute and 'sql' in result and result['sql']:
        sql = result['sql'].strip()
        logger.info(f"Executing SQL: {sql}")
        
        # Basic query validation for security
        sql_lower = sql.lower()
        if any(cmd in sql_lower for cmd in ['insert', 'update', 'delete', 'drop', 'alter', 'create']):
            result['warning'] = "Only read-only queries are allowed for execution"
            result['error'] = "Query contains write operations which are not allowed"
            return result
        
        # Extract the limit from the query to check if it's reasonable
        limit_match = re.search(r'limit\s+(\d+)', sql_lower)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > 1000:  # If limit is too high, enforce a lower limit
                logger.warning(f"Natural language query had excessive LIMIT {limit_value}, reducing to 1000")
                sql = re.sub(r'limit\s+\d+', 'LIMIT 1000', sql, flags=re.IGNORECASE)
                result['sql'] = sql
                result['warning'] = f"Query limit was reduced from {limit_value} to 1000 for performance reasons."
        elif 'limit' not in sql_lower:  # If no limit specified, add a default limit
            if sql.strip().endswith(';'):
                sql = sql[:-1] + " LIMIT 1000;"
            else:
                sql = sql + " LIMIT 1000;"
            result['sql'] = sql
            result['warning'] = "A limit of 1000 rows was added for performance reasons."
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Set a statement timeout (10 seconds)
            cursor.execute("SET statement_timeout TO 10000;")  # 10 seconds
            
            # Execute the query
            logger.info(f"Executing final SQL: {sql}")
            cursor.execute(sql)
            
            # Get the results
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            logger.info(f"Query returned {len(results)} rows")
            result['results'] = results
            result['rows_returned'] = len(results)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing SQL: {error_msg}")
            result['error'] = error_msg
        finally:
            if 'conn' in locals():
                conn.close()
    else:
        logger.warning(f"Query not executed. Execute flag: {execute}, SQL present: {'sql' in result}")
    
    return result

if __name__ == "__main__":
    PORT = int(os.getenv('POSTGRES_SERVER_PORT', args.port))
    logger.info(f"Starting PostgreSQL server on http://localhost:{PORT}")
    logger.info(f"Database config: {DB_CONFIG['host']}:{DB_CONFIG['port']} - {DB_CONFIG['database']}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 