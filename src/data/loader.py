# import sqlite3

# # 1. Connect to the database file
# # If the file does not exist, sqlite3 will create it.
# conn = sqlite3.connect("pdgall-2025-v0.2.1.sqlite")

# # 2. Create a cursor object to interact with the database
# cursor = conn.cursor()

# # 3. Run a query
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# # 4. Fetch results
# tables = cursor.fetchall()
# print("Tables in the database:", tables)

# # Example: read data from a specific table
# cursor.execute(f"SELECT * FROM {tables[0][0]};")

# for _ in range(10):
#     cursor.fetchone

# # 5. Close connection when done
# conn.close()


import sqlite3

db_path = "data/pdgall-2025-v0.2.1.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Show everything in sqlite_master
cursor.execute("SELECT name, type, sql FROM sqlite_master;")
objects = cursor.fetchall()

print("Objects in database:")
for name, type_, sql in objects:
    print(f"- {type_}: {name}")
    if sql:
        print(f"  SQL: {sql[:100]}...")  # preview



