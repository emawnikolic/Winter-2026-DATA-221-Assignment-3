# Winter-2026-DATA-221-Assignment-3

Question 1: Crime Statistics

This script calculates descriptive statistics (mean, median, standard deviation) for crime data to analyze the mathematical center and spread of the dataset. The analysis focuses on how outliers pull the mean away from the median, indicating whether the crime distribution is symmetric or skewed.

Question 2: Data Visualization

This section uses histograms and box plots to visually represent the spread and density of violent crime rates. These plots are used to identify the "shape" of the data and provide a clear visual check for statistical outliers that may not be obvious from raw numbers alone.

Question 3: Data Preprocessing 

This task involves partitioning the kidney disease dataset into distinct training (70%) and testing (30%) sets. This separation is vital to ensure the model is evaluated on "unseen" data, preventing it from simply memorizing the answers (overfitting).

Question 4: KNN Model & Evaluation

A K-Nearest Neighbors model is trained to classify kidney disease, and its performance is measured using a confusion matrix and scores like Precision and Recall. The focus is on understanding why Recall is the most critical metric in a medical context, where missing a sick patient (False Negative) is the most dangerous error.

Question 5: Hyperparameter Tuning

This final script tests multiple values of $k$ ($1, 3, 5, 7, 9$) to find the optimal balance between model complexity and accuracy. The results demonstrate how small $k$ values lead to overfitting (learning noise) while large $k$ values lead to underfitting (oversimplifying the data).
