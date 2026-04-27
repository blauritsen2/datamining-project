# Objective 1: Data Preprocessing: Cleaning, integration, and normalizing movie data from rotten tomatoes and Letterboxd
import re
import os
import pandas as pd
import numpy as np
import nltk

# Helper functions for normalization and text cleaning
def normalize_score(score):
    try:
        if pd.isna(score) or score == "0/0":
            return np.nan
        # Handle fractions (e.g., 3.5/5)
        if "/" in str(score):
            num, den = str(score).split("/")
            return (float(num) / float(den)) * 100
        # Handle Letter Grades (e.g., A-, B+)
        grades = {"A": 95, "B": 75, "C": 55, "D": 35, "F": 15}
        grade = str(score)[0].upper()
        if grade in grades:
            val = grades[grade]
            if "+" in str(score): val += 5
            if "-" in str(score): val -= 5
            return val
    except:
        return np.nan

def clean_review_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace/newlines
    # Regex to replace LS, PS, and other non-standard whitespace
    text = re.sub(r'[\u2028\u2029]', ' ', text)
    return text.strip()

def clean_title(text):
    return str(text).lower().strip() if pd.notna(text) else ""

# Load data
print("Loading raw datasets...")
df_reviews = pd.read_csv('data/raw/rotten_tomatoes_critic_reviews.csv')
df_movies = pd.read_csv('data/raw/rotten_tomatoes_movies.csv')
lb_movies = pd.read_csv('data/raw/movie_data.csv', engine='python', on_bad_lines='warn')
lb_ratings = pd.read_csv('data/raw/ratings_export.csv')

# Merge on the unique movie link
df_merged = pd.merge(df_reviews, df_movies, on="rotten_tomatoes_link", how="inner")

# Drop rows where there is absolutely no text to analyze
df_merged = df_merged.dropna(subset=["review_content"])

# Fill missing scores with a neutral placeholder or handle NaN
df_merged["review_score"] = df_merged["review_score"].fillna("0/0")

# Encode Fresh as 1 and Rotten as 0
df_merged["review_type_encoded"] = df_merged["review_type"].map(
    {"Fresh": 1, "Rotten": 0}
)

# Ensure we drop any rows that couldn't be encoded
df_merged = df_merged.dropna(subset=["review_type_encoded"])

# Apply normalization and text cleaning
df_merged['normalized_review_score'] = df_merged['review_score'].apply(normalize_score)
df_merged['review_content'] = df_merged['review_content'].apply(clean_review_text)

# --- Letterboxd Integration & Master Merge ---
print("Aggregating Letterboxd ratings...")
lb_avg_ratings = lb_ratings.groupby('movie_id').agg({'rating_val': 'mean'}).reset_index()
lb_summarized = pd.merge(lb_movies, lb_avg_ratings, on='movie_id')
lb_summarized['avg_user_rating_100'] = (lb_summarized['rating_val'] / 10.0) * 100

# Extract and standardize release years
df_merged['release_year'] = pd.to_datetime(df_merged['original_release_date'], errors='coerce').dt.year
lb_summarized['year_released'] = pd.to_numeric(lb_summarized['year_released'], errors='coerce')

df_merged = df_merged.dropna(subset=['release_year'])
df_merged['release_year'] = df_merged['release_year'].astype(int)

lb_summarized = lb_summarized.dropna(subset=['year_released'])
lb_summarized['year_released'] = lb_summarized['year_released'].astype(int)

# Clean titles for the final merge
df_merged['clean_title'] = df_merged['movie_title'].apply(clean_title)
lb_summarized['clean_title'] = lb_summarized['movie_title'].apply(clean_title)

# Final Master Merge
print("Performing final data integration...")
master_df = pd.merge(
    df_merged, 
    lb_summarized, 
    left_on=['clean_title', 'release_year'], 
    right_on=['clean_title', 'year_released'], 
    how='inner'
)

# Statistical Filtering (Record Deletion)
review_counts = master_df.groupby(['clean_title', 'release_year']).size().reset_index(name='review_count')
master_df = pd.merge(master_df, review_counts, on=['clean_title', 'release_year'])
master_df = master_df[master_df['review_count'] >= 5]

# Dimensionality Reduction: Selecting essential attributes
essential_cols = [
    'clean_title', 'release_year', 'genres_x', 
    'tomatometer_rating', 'avg_user_rating_100', 
    'normalized_review_score', 'review_content', 
    'top_critic', 'review_type_encoded'
]
master_df = master_df[essential_cols].dropna(subset=['tomatometer_rating', 'avg_user_rating_100'])

# Save Processed Data
os.makedirs("data/processed", exist_ok=True)
output_path = 'data/processed/unified_movie_data.csv'
master_df.to_csv(output_path, index=False)

print("-" * 30)
print(f"SUCCESS: Preprocessing Complete.")
print(f"Final Dataset: {len(master_df):,} rows.")
print(f"Cleaned data saved to: {output_path}")
print("-" * 30)