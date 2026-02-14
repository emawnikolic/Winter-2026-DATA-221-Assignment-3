# Question 3

import pandas as pd
from sklearn.model_selection import train_test_split

kidney_df = pd.read_csv("kidney_disease.csv")

X = kidney_df.drop("classification", axis=1)
y = kidney_df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# RESULT
# Training features shape: (280, 25)
# Testing features shape: (120, 25)
# Training labels shape: (280,) <- one dimensional so there is only one number
# Testing labels shape: (120,)

# COMMENTS:
# We should not train and test a model on the same data because the model
# would simply memorize the training examples instead of learning general patterns.
# This would lead to overfitting and give an unrealistically high accuracy.
# The purpose of the testing set is to evaluate how well the model performs
# on new, unseen data.
# It provides an unbiased estimate of the model’s ability to generalize.
# Splitting the dataset ensures that we properly measure real-world performance.



