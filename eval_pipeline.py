from vector_store import SHLVectorStore
import pandas as pd

# 1. Initialize
print("--- INITIALIZING TEST ENGINE ---")
store = SHLVectorStore()
store.build_index()

# 2. Define Test Queries (Use JDs from the PDF task)
test_queries = [
    "Looking for a Senior Python Developer with experience in AWS and SQL. Needs to lead a small team.",
    "Entry level HR assistant. Focus on communication, scheduling, and people skills.",
    "Data Scientist with expertise in Machine Learning, R, and Python. Behavioral trait: problem solving."
]

print("\n--- STARTING EVALUATION ---")

for i, query in enumerate(test_queries):
    print(f"\nTEST CASE #{i+1}")
    print(f"Input JD: {query}")
    
    # Run the Advanced Logic
    results = store.get_diverse_candidates(query, top_n=5)
    
    print("-" * 30)
    print(f"{'ASSESSMENT NAME':<45} | {'CATEGORY'}")
    print("-" * 30)
    
    for _, row in results.iterrows():
        print(f"{row['name'][:43]:<45} | {row['test_type']}")
    
    # Verify Balance
    types = results['test_type'].value_counts().to_dict()
    print(f"\nSummary of results: {types}")
    print("=" * 60)

print("\nEvaluation Complete. Check terminal for diversity in 'Category' columns.")