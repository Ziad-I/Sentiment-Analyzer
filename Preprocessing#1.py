import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import spacy

nlp = spacy.load('en_core_web_sm')
# Add 'not' to spaCy stop words
spacy_stop_words = spacy.lang.en.stop_words.STOP_WORDS
# add others
words_to_keep = ['not']
for word in words_to_keep:
    spacy_stop_words.discard(word)


# Function to remove stop words from a text using spaCy
def remove_stop_words_spacy(text):
    doc = nlp(text)
    filtered_text = [token.text for token in doc if token.text.lower() not in spacy_stop_words]
    return ' '.join(filtered_text)

df = pd.read_csv("sentimentdataset (Project 1).csv")

df.drop(columns = ['Source', 'ID'], inplace=True)

# Apply the function to the "Message" column
df['Message'] = df['Message'].apply(remove_stop_words_spacy)

print(df['Message'])

# Save the updated DataFrame to a new CSV file
# df.to_csv('path/to/your/new_file_spacy.csv', index=False)