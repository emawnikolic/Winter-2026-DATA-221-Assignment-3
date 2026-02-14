# Question 1

import pandas as pd
import numpy as np

crime_df = pd.read_csv("crime.csv")
violent = crime_df["ViolentCrimesPerPop"]

mean_value = violent.mean()
median_value = violent.median()
std_value = violent.std()
min_value = violent.min()
max_value = violent.max()

# RESULTS
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_value)
print("Minimum:", min_value)
print("Maximum:", max_value)

# COMMENTS:
# If the mean is larger than the median, the distribution is right-skewed,
# meaning there are some high extreme values pulling the mean upward.
# If the mean is smaller than the median, the distribution is left-skewed.
# If they are approximately equal, the distribution is roughly symmetric.
# The mean is more affected by extreme values because it uses all data points
# in its calculation. The median is more robust to outliers because it only
# depends on the middle value of the sorted data.
# Therefore, in the presence of extreme values, the mean changes more than the median.

