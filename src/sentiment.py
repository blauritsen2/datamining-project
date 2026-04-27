# Objective 3: Sentiment Analysis: Analyzing user reviews to determine the sentiment and its impact on movie ratings.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle  # Fix: Import for type hinting


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the preprocessed reviews dataset."""
    return pd.read_csv(filepath)


def plot_sentiment_by_genre(
    df: pd.DataFrame, save_path: str = "../img/sentiment_by_genre.png"
):
    """
    Figure 1: Sentiment Distribution by Genre (Stacked Bar Chart)
    """
    bin_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

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
    ax = data_pct.plot(kind="barh", stacked=True, figsize=(10, 6), color=colors)

    # --- TRUNCATE Y-AXIS LABELS ---
    # Slices labels to 30 chars and adds ellipsis if they exceed that length
    labels = [
        f"{label[:27]}..." if len(label) > 30 else label for label in data_pct.index
    ]
    ax.set_yticklabels(labels)

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
                )

    plt.title("Sentiment Distribution by Genre")
    plt.xlabel("Percentage of Reviews")
    plt.legend(title="Sentiment Bin", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_review_type_distribution(
    df: pd.DataFrame, save_path: str = "review_type_distribution.png"
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
    df: pd.DataFrame, save_path: str = "horizontal_review_boxplot.png"
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


def main():
    data_filepath = "../data/processed/rotten_tomatoes_preprocessed_reviews.csv"
    print(f"Loading data from {data_filepath}...")
    df = load_data(data_filepath)

    print("Generating Figure 1: Sentiment Distribution by Genre...")
    plot_sentiment_by_genre(df)

    print("Generating Figure 2: Review Type Distribution...")
    plot_review_type_distribution(df)

    print("Generating Figure 3: Score Distribution Boxplot...")
    plot_score_distribution(df)

    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
