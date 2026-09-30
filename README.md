# CodeAlpha - Exploratory Data Analysis (Wine Dataset)

**CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)**

## Objective
Explore the classic Wine dataset (178 wines, 13 chemical measurements, 3 classes) by asking
questions first, then answering them with statistics and visualizations.

## Questions asked
1. What does the data look like (size, types, missing values)?
2. Which chemical features differ most between the 3 wine classes?
3. Which features are strongly correlated with each other?
4. Are there outliers?
5. Hypothesis: average alcohol content differs between wine classes.

## Key findings
- **Data quality:** 178 rows, 14 columns, no missing values, no duplicates. Classes are fairly balanced (71 / 59 / 48).
- **Class differences:** class_0 has the highest alcohol (13.74) and proline (1116); class_2 has the lowest flavanoids (0.78) and highest color intensity (7.40); class_1 has the lowest alcohol (12.28) and lowest color intensity (3.09).
- **Correlations:** total_phenols and flavanoids are strongly correlated (0.86); flavanoids also correlate with OD280/OD315 (0.79).
- **Outliers:** a small number (1-4 per feature, IQR rule) in malic_acid, ash, alcalinity_of_ash, magnesium, proanthocyanins, color_intensity and hue. They look like natural variation, so they were kept.
- **Hypothesis test:** one-way ANOVA gives F = 135.08, p < 0.001, so alcohol content differs significantly between wine classes.

## Files
| File | Description |
|---|---|
| `eda_wine.py` | Full EDA script |
| `wine.csv` | Dataset used |
| `01_distributions.png` | Distribution of every feature |
| `02_features_by_class.png` | Key features by wine class |
| `03_correlation_heatmap.png` | Correlation heatmap |
| `04_outliers.png` | Outlier count per feature |

## How to run
```
pip install pandas numpy matplotlib seaborn scipy scikit-learn
python eda_wine.py
```

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy
