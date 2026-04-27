import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load Data ---
print("Loading processed dataset...")
df = pd.read_csv('data/processed/unified_movie_data.csv')

# --- 2. Clean Data ---
print(f"Initial record count: {len(df)}")

df = df.dropna(subset=['tomatometer_rating', 'avg_user_rating_100'])
print(f"Cleaned record count for modeling: {len(df)}")

# --- 3. Feature Engineering ---
# Core feature (your main objective)
features = ['tomatometer_rating']

# OPTIONAL: add more features for comparison (recommended)
if 'top_critic' in df.columns:
    df['top_critic'] = df['top_critic'].astype(int)
    features.append('top_critic')

if 'release_year' in df.columns:
    features.append('release_year')

X = df[features]
y = df['avg_user_rating_100']

print(f"Features used: {features}")

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Models ---

# Linear Regression (baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest (slightly tuned)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# --- 6. Evaluation Function ---
def evaluate(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print(f"\n--- {name} Results ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    return r2

lr_r2 = evaluate("Linear Regression", y_test, lr_preds)
rf_r2 = evaluate("Random Forest", y_test, rf_preds)

# --- 7. Cross-Validation (FIXED: only on training data) ---
print("\nPerforming 5-Fold Cross-Validation...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores):.4f}")

# --- 8. Feature Importance (for Random Forest) ---
if len(features) > 1:
    importance = pd.Series(rf_model.feature_importances_, index=features)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))

    importance.sort_values().plot(kind='barh')
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("report/Figures/feature_importance.png")

# --- 9. Visualization: Predicted vs Actual ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_preds, alpha=0.3, label='Predicted')

# Perfect prediction line
min_val = min(y_test.min(), rf_preds.min())
max_val = max(y_test.max(), rf_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')

plt.title('Random Forest: Predicted vs. Actual Audience Ratings')
plt.xlabel('Actual User Rating (Letterboxd)')
plt.ylabel('Predicted User Rating')
plt.legend()
plt.tight_layout()

plt.savefig('report/Figures/prediction_results.png')

# --- 10. Residual Plot (NEW - great for presentation) ---
residuals = y_test - rf_preds

plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)

plt.title("Residual Distribution (Prediction Errors)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig('report/Figures/residuals.png')

print("\nAll visualizations saved to report/Figures/")