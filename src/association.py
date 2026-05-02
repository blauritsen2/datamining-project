# Objective 2: Association Mining: Identifying frequent patterns and rule-based relationships between professional critic scores and user ratings.
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load the integrated dataset
file_path = 'Data/processed/unified_movie_data.csv'
df = pd.read_csv(file_path)

print("--- Data Distribution Diagnostic ---")
print(f"Total Records: {len(df)}")
print(df[['tomatometer_rating', 'avg_user_rating_100']].describe())
print("-" * 35)

# 2. Discretization (Wider bins to capture more patterns)
df['Critic_Level'] = pd.cut(df['tomatometer_rating'], bins=[0, 50, 60, 100], labels=['Critic_Low', 'Critic_Mid', 'Critic_High'])

# User_Low is bottom 50, High is above 65 (adjusting for 'User Inflation')
df['User_Level'] = pd.cut(df['avg_user_rating_100'], bins=[0, 50, 65, 100], labels=['User_Low', 'User_Mid', 'User_High'])

df['Era'] = pd.cut(df['release_year'], bins=[1900, 1980, 2005, 2026], labels=['Classic', 'Modern', 'Contemporary'])

# 3. Features List
features = df[['genres_x', 'Critic_Level', 'User_Level', 'Era']]
df_encoded = pd.get_dummies(features)

# 4. Run Apriori with 'Deep Search' Support
# Lowering support to 0.001 (0.1%) means a pattern only needs ~800 movies to count.
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)

# 5. Generate Association Rules
# Setting min_threshold to 1.0 to see all relationships that aren't purely random
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 6. Filter for Discrepancy Rules (Disagreement between Critics and Fans)
discrepancy_rules = rules[
    ((rules['antecedents'].apply(lambda x: 'Critic_High' in str(x))) & (rules['consequents'].apply(lambda x: 'User_Low' in str(x)))) |
    ((rules['antecedents'].apply(lambda x: 'Critic_Low' in str(x))) & (rules['consequents'].apply(lambda x: 'User_High' in str(x))))
]

# 7. Output Results
if not discrepancy_rules.empty:
    print(f"Success! Found {len(discrepancy_rules)} discrepancy rules.")
    # Sort by Lift to 
    output = discrepancy_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    print(output.sort_values(by='lift', ascending=False).head(10))
else:
    print("No discrepancies found at 0.1% support. This suggests high agreement in the dataset.")
    print("Top 5 general agreement rules (for your 'Results' slide):")
    agreement_rules = rules[rules['lift'] > 1.2].sort_values(by='confidence', ascending=False)
    print(agreement_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))