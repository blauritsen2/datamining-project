# Objective 3: Sentiment Analysis: Analyzing user reviews to determine the sentiment and its impact on movie ratings.

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Rectangle  # Fix: Import for type hinting
from transformers import pipeline

os.makedirs("report/Figures", exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the preprocessed reviews dataset."""
    return pd.read_csv(filepath)


def plot_sentiment_by_genre(
    df: pd.DataFrame, save_path: str = "report/Figures/sentiment_by_genre.png"
):
    """
    Figure 1: Sentiment Distribution by Genre (Stacked Bar Chart)
    """
    bin_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    os.makedirs("report/Figures", exist_ok=True)

    df_plot = df.copy()
    df_plot["bins"] = pd.cut(
        df_plot["normalized_review_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    top_genres = df_plot["genres"].value_counts().nlargest(10).index

    data = (
        df_plot[df_plot["genres"].isin(top_genres)]
        .groupby(["genres", "bins"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    data_pct = data.div(data.sum(axis=1), axis=0)

    # Plotting
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    ax = data_pct.plot(kind="barh", stacked=True, figsize=(12, 8), color=colors)

    # --- TRUNCATE Y-AXIS LABELS ---
    # Slices labels to 30 chars and adds ellipsis if they exceed that length
    labels = [
        f"{label[:27]}..." if len(label) > 30 else label for label in data_pct.index
    ]
    ax.set_yticklabels(labels, fontsize=11)

    # Labels - Fixed the Pylance attribute warnings via type narrowing
    for p in ax.patches:
        if isinstance(p, Rectangle):
            width = p.get_width()
            if width > 0.05:
                x = p.get_x() + width / 2
                y = p.get_y() + p.get_height() / 2
                ax.annotate(
                    f"{width:.0%}",
                    (x, y),
                    ha="center",
                    va="center",
                    color="black",
                    weight="bold",
                    fontsize=10,
                )

    plt.title("Sentiment Distribution by Genre")
    plt.xlabel("Percentage of Reviews")
    plt.legend(title="Sentiment Bin", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_review_type_distribution(
    df: pd.DataFrame, save_path: str = "report/Figures/review_type_distribution.png"
):
    """
    Figure 2: Review Type Distribution
    """
    plt.figure()
    counts = df["review_type"].value_counts()
    labels = counts.index
    colors = ["#60eb5e", "#f6765c"]

    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title("Review Type Distribution")
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_score_distribution(
    df: pd.DataFrame, save_path: str = "report/Figures/horizontal_review_boxplot.png"
):
    """
    Figure 3: Horizontal Boxplot - Score Distribution: Fresh vs. Rotten
    """
    # 1. Filter the data (ensuring we work on a copy to avoid warnings)
    x_filtered = df[df["normalized_review_score"] <= 1.0].copy()

    # 2. Create the Horizontal Boxplot
    plt.figure(figsize=(10, 5))

    # Swapping x and y:
    # x becomes the numeric score, y becomes the category
    sns.boxplot(
        data=x_filtered,
        x="normalized_review_score",
        y="review_type",
        palette={"Fresh": "green", "Rotten": "red"},  # Matching the RT theme
        hue="review_type",
        legend=False,
    )

    # 3. Customizing the Axis and Labels
    plt.title("Score Distribution: Fresh vs. Rotten", fontsize=14)
    plt.xlabel("Normalized Review Score (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Review Type", fontsize=12)

    # Adding a vertical grid to help see score thresholds
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # 4. Save the plot
    plt.savefig(save_path)
    plt.close()


def apply_bert_sentiment(df: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
    """
    Applies BERT sentiment classification to a sample of the dataset and returns the exploded dataframe.
    """
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True,
        device=device,
        batch_size=64,
    )

    df_sample = df.sample(sample_size, random_state=42).copy()
    texts = df_sample["clean_review"].astype(str).tolist()

    print(f"Inference on {'GPU' if device == 0 else 'CPU'}...")
    results = classifier(texts, truncation=True, max_length=512)

    # Updated score extraction logic
    df_sample["bert_sentiment_score"] = [
        next(label["score"] for label in r if label["label"] == "POSITIVE")
        if isinstance(r, list)
        else (r["score"] if r["label"] == "POSITIVE" else 1 - r["score"])
        for r in results
    ]

    df_exploded = df_sample.explode("genres")

    df_exploded["bert_sentiment_score_bins"] = pd.cut(
        df_exploded["bert_sentiment_score"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
    )

    return df_exploded


def plot_bert_sentiment_distribution(
    df_exploded: pd.DataFrame,
    save_path: str = "report/Figures/genre_sentiment_distribution.png",
):
    """
    Figure 4: Emotional Profiles by Genre using BERT Sentiment Distribution
    """
    sentiment_counts = (
        df_exploded.groupby(["genres", "bert_sentiment_score_bins"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    top_10_genres = df_exploded["genres"].value_counts().nlargest(10).index
    sentiment_counts_filtered = sentiment_counts.loc[top_10_genres]
    sentiment_pct = sentiment_counts_filtered.div(
        sentiment_counts_filtered.sum(axis=1), axis=0
    )

    plt.figure()
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    ax = sentiment_pct.plot(
        kind="barh", stacked=True, figsize=(12, 8), color=colors, width=0.8
    )

    plt.title("Emotional Profiles by Genre: BERT Sentiment Distribution", fontsize=15)
    plt.xlabel("Percentage of Reviews", fontsize=12)
    plt.ylabel("Genre", fontsize=12)
    plt.legend(title="Sentiment Bin", bbox_to_anchor=(1.05, 1), loc="upper left")

    for p in ax.patches:
        if isinstance(p, Rectangle):
            width = p.get_width()
            if width > 0.05:
                ax.annotate(
                    f"{width:.0%}",
                    (p.get_x() + width / 2, p.get_y() + p.get_height() / 2),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def export_bert_anomalies(
    df_exploded: pd.DataFrame,
    num_samples: int = 10,
    output_path: str = "report/Figures/bert_anomalies_sample.txt",
):
    """
    Extracts random anomalous reviews (e.g., Fresh but Very Negative) and writes them to a file.
    """
    anomalies = df_exploded[
        (df_exploded["review_type"] == "Fresh")
        & (df_exploded["bert_sentiment_score_bins"] == "Very Negative")
    ]

    sample_reviews = anomalies["clean_review"].dropna().unique()

    if len(sample_reviews) > 0:
        sample_size = min(num_samples, len(sample_reviews))
        selected_samples = np.random.choice(
            sample_reviews, size=sample_size, replace=False
        )
    else:
        selected_samples = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Sample of 'Fresh' reviews classified as 'Very Negative' by BERT:\n")
        f.write("-" * 60 + "\n")
        for i, review in enumerate(selected_samples, 1):
            f.write(f"{i}. {review}\n")
            f.write("-" * 60 + "\n")
    print(f"Exported {len(selected_samples)} anomalous reviews to {output_path}")


def main():
    data_filepath = "data/processed/rotten_tomatoes_preprocessed_reviews.csv"
    print(f"Loading data from {data_filepath}...")
    df = load_data(data_filepath)

    print("Generating Figure 1: Sentiment Distribution by Genre...")
    plot_sentiment_by_genre(df)

    print("Generating Figure 2: Review Type Distribution...")
    plot_review_type_distribution(df)

    print("Generating Figure 3: Score Distribution Boxplot...")
    plot_score_distribution(df)

    print("Applying BERT sentiment analysis...")
    df_exploded = apply_bert_sentiment(df)

    print("Generating Figure 4: BERT Sentiment Distribution...")
    plot_bert_sentiment_distribution(df_exploded)

    print("Exporting BERT anomalies to text file...")
    export_bert_anomalies(df_exploded)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
