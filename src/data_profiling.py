import pandas as pd
import os

# Define your file paths
files = {
    "RT Movies (Raw)": "data/raw/rotten_tomatoes_movies.csv",
    "RT Reviews (Raw)": "data/raw/rotten_tomatoes_critic_reviews.csv",
    "Letterboxd Movies (Raw)": "data/raw/movie_data.csv",
    "Letterboxd Ratings (Raw)": "data/raw/ratings_export.csv",
    "Unified Movie Data (Processed)": "data/processed/unified_movie_data.csv"
}

print("="*50)
print("DATASET DIMENSIONS & ATTRIBUTE PROFILING")
print("="*50)

for name, path in files.items():
    if os.path.exists(path):
        # engine='python' handles the malformed lines in your movie_data.csv
        df = pd.read_csv(path, engine='python', on_bad_lines='warn')
        
        rows, cols = df.shape
        
        print(f"\n[ {name} ]")
        print(f"Total Records (Rows): {rows:,}")
        print(f"Total Attributes (Cols): {cols}")
        print(f"Attributes: {', '.join(df.columns.tolist()[:10])}...") # Shows first 10 for brevity
    else:
        print(f"\n[ {name} ] - File not found at {path}")

print("\n" + "="*50)