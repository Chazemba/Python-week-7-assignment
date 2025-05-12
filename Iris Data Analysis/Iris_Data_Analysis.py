# Iris Dataset Analysis Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for plots
sns.set_style("whitegrid")

# Load the iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Inspect the data
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Clean the data
df_cleaned = df.dropna()

# Basic statistics
print("\nDescriptive statistics:")
print(df_cleaned.describe())

# Grouping by target
grouped_mean = df_cleaned.groupby("target")["sepal length (cm)"].mean()
print("\nAverage Sepal Length by Species:")
print(grouped_mean)

# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(df_cleaned.index, df_cleaned["sepal length (cm)"], color='green')
plt.title("Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
grouped_mean.plot(kind="bar", color="skyblue")
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Avg Sepal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(6, 4))
sns.histplot(df_cleaned["petal length (cm)"], bins=20, kde=True, color="orchid")
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_cleaned,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target",
    palette="deep"
)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()