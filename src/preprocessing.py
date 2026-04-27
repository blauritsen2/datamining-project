import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 0. Setup NLTK (For Objective 3) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- 1. Load Data ---
# Using engine='python' to handle malformed lines [cite: 26]
print("Loading the datasets...")
rt_movies = pd.read_csv('data/raw/rotten_tomatoes_movies.csv')
rt_reviews = pd.read_csv('data/raw/rotten_tomatoes_critic_reviews.csv')
lb_movies = pd.read_csv('data/raw/movie_data.csv', engine='python', on_bad_lines='warn')
lb_ratings = pd.read_csv('data/raw/ratings_export.csv')

# --- 2. Extract & Standardize Years ---
# Converting raw dates to discrete year integers to ensure consistency [cite: 111]
rt_movies['release_year'] = pd.to_datetime(rt_movies['original_release_date'], errors='coerce').dt.year
rt_movies = rt_movies.dropna(subset=['release_year'])
rt_movies['release_year'] = rt_movies['release_year'].astype(int)
lb_movies = lb_movies.dropna(subset=['year_released'])
lb_movies['year_released'] = lb_movies['year_released'].astype(int)

# --- 3. Clean Titles for Matching ---
def clean_title(text):
    return str(text).lower().strip() if pd.notna(text) else ""

rt_movies['clean_title'] = rt_movies['movie_title'].apply(clean_title)
lb_movies['clean_title'] = lb_movies['movie_title'].apply(clean_title)

# --- 4. Intermediate Aggregation (The Memory Saver) ---
lb_avg_ratings = lb_ratings.groupby('movie_id').agg({'rating_val': 'mean'}).reset_index()
lb_summarized = pd.merge(lb_movies, lb_avg_ratings, on='movie_id')
rt_combined = pd.merge(rt_movies, rt_reviews, on='rotten_tomatoes_link')

# --- 5. Corrected Normalization ---
# Correcting scale from 1-10 (SamLearner dataset) to 0-100 to prevent bias [cite: 282-284]
lb_summarized['avg_user_rating_100'] = (lb_summarized['rating_val'] / 10.0) * 100

# --- 6. Final Integration (The Master Merge) ---
master_df = pd.merge(
    rt_combined, 
    lb_summarized, 
    left_on=['clean_title', 'release_year'], 
    right_on=['clean_title', 'year_released'], 
    how='inner'
)

# --- 7. Statistical Filtering (Record Deletion) ---
# Removing 'insignificant' movies with low review counts to ensure data quality 
review_counts = master_df.groupby(['clean_title', 'release_year']).size().reset_index(name='review_count')
master_df = pd.merge(master_df, review_counts, on=['clean_title', 'release_year'])
master_df = master_df[master_df['review_count'] >= 5]

# --- 8. Dimensionality Reduction & Multicollinearity ---
# Selecting only essential attributes to reduce high-dimensionality 
essential_cols = [
    'clean_title', 'release_year', 'genres_x', 
    'tomatometer_rating', 'avg_user_rating_100', 
    'review_content', 'top_critic'
]
master_df = master_df[essential_cols]

# Checking for redundant feature correlation before modeling 
multicol_check = master_df['tomatometer_rating'].corr(master_df['avg_user_rating_100'])
print(f"Internal Feature Multicollinearity Check: {multicol_check:.2f}")

def sanitize_text(text):
    if pd.isna(text):
        return ""
    # Regex to replace LS, PS, and other non-standard whitespace
    return re.sub(r'[\u2028\u2029]', ' ', str(text))

master_df['review_content'] = master_df['review_content'].apply(sanitize_text)

# --- 9. Save Processed Data ---
output_path = 'data/processed/unified_movie_data.csv'
master_df.to_csv(output_path, index=False)

print("-" * 30)
print(f"SUCCESS: Preprocessing Complete.")
print(f"Final Dataset: {len(master_df)} rows x {len(master_df.columns)} attributes.")
print(f"Cleaned data saved to: {output_path}")
print("-" * 30)