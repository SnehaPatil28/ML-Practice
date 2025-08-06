from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/tested.csv')
df.head()

df.info()
df.describe()
df.isnull().sum()

df.drop(['Cabin'], axis=1, inplace=True)
df.head()

# Replace null values in 'Age' column with the mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.head()

# Replace null values in 'Age' column with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.isnull().sum()

# Replace null values in 'Fare' column with the mean
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df.isnull().sum()

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df = pd.get_dummies(df, columns=['Age'], drop_first=True)

# Display the first few rows of the updated DataFrame
display(df.head())

# Verify that all null values have been imputed and new columns created
df.isnull().sum()

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
# We will drop 'PassengerId', 'Name', and 'Ticket' as they are unlikely to be useful features for prediction
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
y = df['Survived']

print(X)
print(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape (X_train, y_train):", X_train.shape, y_train.shape)
print("Testing set shape (X_test, y_test):", X_test.shape, y_test.shape)
