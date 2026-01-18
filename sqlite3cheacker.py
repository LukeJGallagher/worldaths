import sqlite3
import pandas as pd

# Path to the DB file
db_path = r"C:\Users\l.gallagher\Documents\Performance Analysis\Sport Detailed Data\Athletics\Sport\world_athletics\SQL\ksa_modal_results_men.db"

# Connect and read everything
with sqlite3.connect(db_path) as conn:
    # Step 1: Show available tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("📋 Available tables in DB:\n", tables)

    # Step 2: Load all rows from the first table
    if not tables.empty:
        table_name = tables.iloc[0, 0]  # Use first table name
        print(f"\n📄 Reading from table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(df)
    else:
        print("❌ No tables found in database.")
