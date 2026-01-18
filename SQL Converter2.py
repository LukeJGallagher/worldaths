import os
import pandas as pd
import sqlite3

# Input and output folders
input_folder = r"Data/"
output_folder = r"SQL/"
os.makedirs(output_folder, exist_ok=True)

# Function to infer SQLite types
def infer_sqlite_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "REAL"
    else:
        return "TEXT"

# Stats counters
processed_count = 0
error_count = 0

# Loop through all CSV files
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)
        base_name = os.path.splitext(file)[0].replace(" ", "_").lower()
        table_name = base_name
        db_path = os.path.join(output_folder, f"{base_name}.db")

        print(f"\n📦 Processing: {file}")

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"⚠️ Skipping empty file: {file}")
                continue

            # Connect to new SQLite DB
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Build CREATE TABLE with inferred types
            columns = []
            for col in df.columns:
                col_type = infer_sqlite_type(df[col])
                columns.append(f"`{col}` {col_type}")
            create_stmt = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns)});"
            cursor.execute(create_stmt)

            # Insert data
            df.to_sql(table_name, conn, if_exists='append', index=False)

            conn.commit()
            conn.close()

            print(f"✅ Done: {file} → {db_path}")
            processed_count += 1

        except Exception as e:
            print(f"❌ Error with {file}: {e}")
            error_count += 1
            continue

print(f"\n🏁 Finished. Processed: {processed_count}, Errors: {error_count}")
