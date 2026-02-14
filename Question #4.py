# Question 4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# RESULTS
# Confusion Matrix:
#  [[39  0]
#  [ 2 20]]
# Accuracy:  0.9672
# Precision: 1.0000
# Recall:    0.9091

# COMMENTS:
# In the context of kidney disease, a True Positive means correctly identifying a sick patient,
# while a True Negative means correctly identifying a healthy one.
# A False Positive is a 'false alarm' where a healthy person is flagged,
# whereas a False Negative is a dangerous error where a sick patient is missed.
# Accuracy alone is insufficient because if the dataset is imbalanced,
# the model could look successful while failing to detect the actual minority of sick patients.
# Recall is the most critical metric here because the cost of missing a kidney disease case
# (a False Negative) is much higher than the cost of a false alarm.
# High recall ensures that we catch as many true cases as possible for early intervention.