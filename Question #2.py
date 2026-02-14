# Question 2

import pandas as pd
import matplotlib.pyplot as plt

crime_df = pd.read_csv("crime.csv")
violent = crime_df["ViolentCrimesPerPop"]


plt.figure()
plt.hist(violent, bins=20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(violent)
plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Values")
plt.show()


# COMMENTS:
# The histogram shows how the values of violent crimes per population
# are distributed across different ranges. It helps us see whether the
# data is concentrated in certain intervals or spread out widely.
# If the bars are uneven and have a long tail on one side, the data is skewed.
# The box plot shows the median as the line inside the box.
# The length of the box represents the interquartile range, showing how spread
# out the middle 50% of the data is.
# If there are points beyond the whiskers, the box plot suggests the presence
# of outliers in the dataset.

