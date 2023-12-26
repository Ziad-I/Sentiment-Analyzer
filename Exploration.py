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

#bar chart for sources and target

# Group by 'source' and 'target' and get the count
grouped_data = df.groupby(['Source', 'Target']).size().unstack()

# Plotting the bar chart
fig, ax = plt.subplots()
width = 0.25  # the width of the bars

sources = grouped_data.index
ind = range(len(sources))

bar_0 = ax.bar(ind, grouped_data[0], width, label='0')
bar_1 = ax.bar([i + width for i in ind], grouped_data[1], width, label='1')

# Adding labels and title
ax.set_xlabel('Source')
ax.set_ylabel('Count')
ax.set_title('Count of Positive and Negative by Source')
ax.set_xticks([i + width/2 for i in ind])
ax.set_xticklabels(sources)
ax.legend()

for bar in bar_0:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

for bar in bar_1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')


plt.show()