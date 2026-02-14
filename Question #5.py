# Question 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("kidney_disease.csv")
df['classification'] = df['classification'].str.replace(r'\t', '', regex=True).str.strip()
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype('category').cat.codes

df.dropna(inplace=True)

X = df.drop("classification", axis=1)
y = df["classification"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"k": k, "Accuracy": acc})

results_df = pd.DataFrame(results)
print("K-Value vs Accuracy Table:")
print(results_df.to_string(index=False))

best_k = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\nThe value of k that gives the highest accuracy is k={int(best_k['k'])} with {best_k['Accuracy']:.4f}")

# RESULTS
# K-Value vs Accuracy Table:
#  k  Accuracy
#  1  1.000000
#  3  1.000000
#  5  0.967213
#  7  0.950820
#  9  0.934426
#
# The value of k that gives the highest accuracy is k=1 with 1.0000

# COMMENTS:
# Changing the value of k adjusts the 'smoothness' of the decision boundary between classes.
# Very small values of k (like k=1) make the model highly sensitive to noise or outliers in the training data, leading to overfitting where the model
# learns specific quirks rather than general patterns.
# Conversely, very large values of k can cause underfitting because the model takes too many neighbors into account, potentially including data points
# from the opposite class and 'blurring' the distinction.
# Finding the right k is a balance between capturing the local data structure and maintaining a boundary that generalizes well to new, unseen patients.