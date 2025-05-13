import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset (you can replace this URL with your own dataset path)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
try:
    df = pd.read_csv(url, header=None, names=column_names)
except FileNotFoundError:
    print("File not found, please check the path.")
except pd.errors.EmptyDataError:
    print("The file is empty, please check the contents.")
except Exception as e:
    print(f"An error occurred: {e}")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nData Types and Missing Values:")
print(df.info())
print(df.isnull().sum())

# Clean the dataset by dropping missing values (or use df.fillna() to fill missing data)
df.dropna(inplace=True)


# Compute basic statistics for the numerical columns
print("\nBasic Statistics for Numerical Columns:")
print(df.describe())

# Group by the 'species' column and compute the mean for each group
print("\nGroup by 'species' and compute the mean for each group:")
grouped = df.groupby('species').mean()
print(grouped)


# Set the plot style to 'seaborn' for better aesthetics
sns.set(style="whitegrid")

# Line chart showing trends (For simplicity, using 'sepal_length' here)
plt.figure(figsize=(10,6))
plt.plot(df['sepal_length'], label='Sepal Length', color='blue')
plt.title('Sepal Length Over Time')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.legend()
plt.show()

# Bar chart comparing the average petal length per species
plt.figure(figsize=(8,6))
df.groupby('species')['petal_length'].mean().plot(kind='bar', color='green')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.show()

# Histogram of 'sepal_length' distribution
plt.figure(figsize=(8,6))
df['sepal_length'].hist(bins=20, edgecolor='black', color='purple')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Scatter plot to visualize the relationship between 'sepal_length' and 'petal_length'
plt.figure(figsize=(8,6))
plt.scatter(df['sepal_length'], df['petal_length'], color='red')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Additional visualization using seaborn: Pairplot for all numerical columns
sns.pairplot(df, hue='species')
plt.show()
