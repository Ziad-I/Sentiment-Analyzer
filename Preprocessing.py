import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
# Add 'not' to spaCy stop words
spacy_stop_words = spacy.lang.en.stop_words.STOP_WORDS

# add others
words_to_keep = ["not", "no", "never", "but", "only", "against",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "won't", "wouldn't", "can't", "cannot",
    "could've", "should've", "would've", "doesn't", "didn't", "isn't", "ain't"]

for word in words_to_keep:
    spacy_stop_words.discard(word)

# Function for lemmatization and removing stop words
def lemmatize_and_remove_stop_words(text):
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc if token.text.lower() not in spacy_stop_words]
    return ' '.join(lemmatized_text)

df = pd.read_csv("sentimentdataset (Project 1).csv")

df.drop(columns = ['Source', 'ID'], inplace=True)

# Apply the function to the "Message" column
df['Message'] = df['Message'].apply(lemmatize_and_remove_stop_words)

print(df['Message'])

# Save the updated DataFrame to a new CSV file
df.to_csv('sentimentdataset_stopwords_lemmatized.csv', index=False)