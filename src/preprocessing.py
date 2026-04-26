# Objective 1: Data Preprocessing: Cleaning, integration, and normalizing movie data from rotten tomatoes and Letterboxd

import re
import os
import pandas as pd
import numpy as np

# Load data
df_reviews = pd.read_csv("data/rotten_tomatoes/rotten_tomatoes_critic_reviews.csv")
df_movies = pd.read_csv("data/rotten_tomatoes/rotten_tomatoes_movies.csv")

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


def normalize_score(score):
    try:
        if pd.isna(score) or score == "0/0":
            return np.nan
        # Handle fractions (e.g., 3.5/5)
        if "/" in str(score):
            num, den = score.split("/")
            return float(num) / float(den)
        # Handle Letter Grades (e.g., A-, B+)
        grades = {"A": 0.95, "B": 0.75, "C": 0.55, "D": 0.35, "F": 0.15}
        grade = str(score)[0].upper()
        if grade in grades:
            val = grades[grade]
            if "+" in str(score):
                val += 0.05
            if "-" in str(score):
                val -= 0.05
            return val
    except:
        return np.nan


df_merged["normalized_review_score"] = df_merged["review_score"].apply(normalize_score)
df_merged = df_merged[df_merged["normalized_review_score"] <= 1.0]


def clean_review_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace/newlines
    return text.strip()


df_merged["clean_review"] = df_merged["review_content"].apply(clean_review_text)

os.makedirs("data/processed", exist_ok=True)
df_merged.to_csv("data/processed/rotten_tomatoes_preprocessed_reviews.csv", index=False)
