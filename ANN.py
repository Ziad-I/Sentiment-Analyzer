import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras import layers, Sequential
from scikeras.wrappers import KerasClassifier
from matplotlib import pyplot
import spacy
import matplotlib.pyplot as plt

df = pd.read_csv("sentimentdataset (Project 1).csv")
df = df.drop(["Source", "ID"], axis=1)

features = ["Message"]
targets = ["Target"]

nlp = spacy.load("en_core_web_sm")

custom_stop_words = set(nlp.Defaults.stop_words) - {
    "not", "no", "never", "but", "only", "against",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "won't", "wouldn't", "can't", "cannot",
    "could've", "should've", "would've", "doesn't", "didn't", "isn't", "ain't"
}

def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text.lower() not in custom_stop_words]
    return ' '.join(tokens)

df["Message"] = df["Message"].apply(process_text)

X = df.iloc[:, 0]
y = df.iloc[:, -1]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Message'])
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df['Message'])


print("\nTf-idf Feature Names:")
a = vectorizer.get_feature_names_out()
print(a)

# Display the Tf-idf matrix
print("\nTf-idf Matrix:")
b = tfidf_matrix.toarray()
print(b)


print("CountVectorizer Feature Names:")
print(count_vectorizer.get_feature_names_out())
print("\nCountVectorizer Matrix:")
print(count_matrix.toarray())


# counts = y.value_counts()
# pyplot.bar(x=["0", "1"], height=[counts[0], counts[1]])
# pyplot.show()

# print(X)
