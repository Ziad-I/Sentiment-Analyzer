import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("sentimentdataset (Project 1).csv")

source_counts = df['Source'].value_counts()
target_counts = df['Target'].value_counts()

# # Plot source as a bar chart
# plt.figure(figsize=(8, 6))
# source_counts.plot(kind='bar', color='skyblue')
# plt.title('Frequency of Each Category in the "Source" Column')
# plt.xlabel('Source')
# plt.ylabel('Frequency')
# plt.show()

# Plot source as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Distribution of Categories in the "Source" Column')
plt.show()

# # Plot target as a bar chart
# plt.figure(figsize=(8, 6))
# target_counts.plot(kind='bar', color='skyblue')
# plt.title('Frequency of Each Category in the "Target" Column')
# plt.xlabel('Target')
# plt.ylabel('Frequency')
# plt.show()

# Plot target as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Distribution of Categories in the "Target" Column')
plt.show()