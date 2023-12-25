import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt

df = pd.read_csv("sentimentdataset (Project 1).csv")

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