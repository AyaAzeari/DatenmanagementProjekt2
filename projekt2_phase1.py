import numpy as np
import psycopg2
import time

# Data generator for matrices with sparsity
def generate_matrices(l, sparsity):
    m = l - 1
    n = l + 1
    A = np.random.rand(m, l)
    B = np.random.rand(l, n)
    A[np.random.rand(m, l) < sparsity] = 0
    B[np.random.rand(l, n) < sparsity] = 0
    return A, B

# Import matrices into the database
def import_matrices_to_db(A, B, connection):
    cur = connection.cursor()
    cur.execute("DROP TABLE IF EXISTS A, B")
    cur.execute("CREATE TABLE A (i INT, j INT, val DOUBLE PRECISION)")
    cur.execute("CREATE TABLE B (i INT, j INT, val DOUBLE PRECISION)")
    
    # Insert data into tables
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                cur.execute("INSERT INTO A (i, j, val) VALUES (%s, %s, %s)", (i + 1, j + 1, float(A[i, j])))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if B[i, j] != 0:
                cur.execute("INSERT INTO B (i, j, val) VALUES (%s, %s, %s)", (i + 1, j + 1, float(B[i, j])))

    connection.commit()
    cur.close()

# Ansatz 0: Client-side matrix multiplication without sub-cubic algorithms
def ansatz0(A, B):
    m, l = A.shape
    l, n = B.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(l):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Ansatz 1: Example 2.1 - Matrix Multiply with Sparse Representation
def ansatz1(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT A.i, B.j, SUM(A.val * B.val) AS product
        FROM A, B
        WHERE A.j = B.i
        GROUP BY A.i, B.j;
    """)
    result = cur.fetchall()
    cur.close()
    return result

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=postgres user=postgres password=your_password host=localhost port=5432")

# 1. Generate matrices
l = 4
sparsity = 0.5
A, B = generate_matrices(l, sparsity)

# Print generated matrices
print("Generated Matrix A:")
print(A)
print("\nGenerated Matrix B:")
print(B)

# 2. Import generated matrices into the database
import_matrices_to_db(A, B, conn)

# Verify the imported matrices
def verify_imported_matrices(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM A ORDER BY i, j")
    rows_A = cur.fetchall()
    cur.execute("SELECT * FROM B ORDER BY i, j")
    rows_B = cur.fetchall()
    cur.close()
    
    # Convert query results back to matrices
    m, l = A.shape
    _, n = B.shape
    A_db = np.zeros((m, l))
    B_db = np.zeros((l, n))
    
    for (i, j, val) in rows_A:
        A_db[i-1, j-1] = val
    
    for (i, j, val) in rows_B:
        B_db[i-1, j-1] = val

    print("\nMatrix A from Database:")
    print(A_db)
    print("\nMatrix B from Database:")
    print(B_db)
    
    # Compare the original and database matrices
    assert np.allclose(A, A_db), "Matrix A in database does not match the generated matrix"
    assert np.allclose(B, B_db), "Matrix B in database does not match the generated matrix"
    print("\nVerification successful: Imported matrices match the generated matrices.")

# Verify the imported matrices
verify_imported_matrices(conn)

# Toy Example Matrices
toyA = np.array([[3, 2, 0], [1, 0, 2]])
toyB = np.array([[1, 2], [0, 1], [4, 0]])

# Import Toy matrices into the database
import_matrices_to_db(toyA, toyB, conn)

# Function to execute and verify matrix multiplication
def verify_matrix_multiplication(toyA, toyB, conn):
    # Ansatz 0: Client-side multiplication
    C_0 = ansatz0(toyA, toyB)
    print("\nClient-side Toy Matrix Multiplication:")
    print(C_0)
    
    # Ansatz 1: SQL-based multiplication
    result_1 = ansatz1(conn)
    print("\nSQL-based Toy Matrix Multiplication:")
    for row in result_1:
        print(f"Row: {row}")
    
    # Convert SQL result to matrix format for comparison
    m, n = toyA.shape[0], toyB.shape[1]
    C_sql = np.zeros((m, n))
    for (i, j, val) in result_1:
        C_sql[i-1, j-1] = val
    print("\nSQL-based Result Matrix:")
    print(C_sql)
    
    # Compare the results
    assert np.allclose(C_0, C_sql), "Client-side result does not match SQL-based result"
    print("\nVerification successful: Client-side and SQL-based results match.")

# Verify Toy Example
verify_matrix_multiplication(toyA, toyB, conn)

conn.close()
