"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Dataset: Wine dataset (178 wines, 13 chemical measurements, 3 cultivars)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_wine

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Questions we want to answer before touching the data
# ---------------------------------------------------------------
QUESTIONS = """
Q1. What does the data look like (size, types, missing values)?
Q2. Which chemical features differ most between the 3 wine classes?
Q3. Which features are strongly correlated with each other?
Q4. Are there outliers / anomalies?
Q5. Hypothesis: average alcohol content differs between wine classes.
"""
print(QUESTIONS)

# ---------------------------------------------------------------
# 2. Load data and save as CSV
# ---------------------------------------------------------------
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["wine_class"] = df["target"].map(dict(enumerate(["class_0", "class_1", "class_2"])))
df = df.drop(columns="target")
df.to_csv("wine.csv", index=False)

# ---------------------------------------------------------------
# 3. Data structure
# ---------------------------------------------------------------
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass balance:\n", df["wine_class"].value_counts())
print("\nSummary statistics:\n", df.describe().T.round(2))

features = [c for c in df.columns if c != "wine_class"]

# ---------------------------------------------------------------
# 4. Distributions of all features
# ---------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(df[col], kde=True, ax=ax, color="#4c72b0")
    ax.set_title(col)
for ax in axes.ravel()[len(features):]:
    ax.axis("off")
plt.suptitle("Distribution of each feature", fontsize=16)
plt.tight_layout()
plt.savefig("01_distributions.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. Feature vs wine class (Q2)
# ---------------------------------------------------------------
key = ["alcohol", "flavanoids", "color_intensity", "proline"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.ravel(), key):
    sns.boxplot(data=df, x="wine_class", y=col, hue="wine_class",
                palette="Set2", legend=False, ax=ax)
    ax.set_title(f"{col} by wine class")
plt.tight_layout()
plt.savefig("02_features_by_class.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Correlation (Q3)
# ---------------------------------------------------------------
corr = df[features].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7})
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("03_correlation_heatmap.png", dpi=120)
plt.close()

pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\nTop 5 strongest correlations:\n",
      pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).round(2))

# ---------------------------------------------------------------
# 7. Outliers using the IQR rule (Q4)
# ---------------------------------------------------------------
outliers = {}
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    if n:
        outliers[col] = int(n)
print("\nOutlier count (IQR rule):", outliers)

plt.figure(figsize=(9, 5))
plt.bar(outliers.keys(), outliers.values(), color="#dd8452")
plt.xticks(rotation=60, ha="right")
plt.ylabel("Number of outliers")
plt.title("Outliers per feature (IQR rule)")
plt.tight_layout()
plt.savefig("04_outliers.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 8. Hypothesis test (Q5): ANOVA on alcohol across classes
# ---------------------------------------------------------------
groups = [g["alcohol"].values for _, g in df.groupby("wine_class")]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\nANOVA alcohol ~ wine_class: F = {f_stat:.2f}, p = {p_val:.2e}")
print("Result:", "Reject H0 - alcohol differs between classes"
      if p_val < 0.05 else "Fail to reject H0")

print("\nMean of each feature per class:\n",
      df.groupby("wine_class")[features].mean().T.round(2))
