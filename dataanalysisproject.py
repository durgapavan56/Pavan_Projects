import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a built-in dataset from seaborn
df = sns.load_dataset('iris')  # You can change 'iris' to another seaborn dataset like 'titanic'

# Display the first few rows of the dataset
print("Dataset loaded successfully. Here's a preview:")
print(df.head())

# Plot a histogram of one of the columns (e.g., 'sepal_length')
sns.histplot(df['sepal_length'], kde=True)  # Histogram with a Kernel Density Estimate (KDE)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Create a pairplot to show relationships between columns
sns.pairplot(df)  # Pairplot to visualize relationships between all numeric columns
plt.show()

# Create a heatmap of the correlations between columns
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])  
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
