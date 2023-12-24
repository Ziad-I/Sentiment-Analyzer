import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# nlp = spacy.load('en_core_web_sm')
# # Add 'not' to spaCy stop words
# spacy_stop_words = spacy.lang.en.stop_words.STOP_WORDS

# # add others
# words_to_keep = ["not", "no", "never", "but", "only", "against",
#     "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
#     "hasn't", "haven't", "hadn't", "won't", "wouldn't", "can't", "cannot",
#     "could've", "should've", "would've", "doesn't", "didn't", "isn't", "ain't"]

# for word in words_to_keep:
#     spacy_stop_words.discard(word)

# # Function for lemmatization and removing stop words
# def lemmatize_and_remove_stop_words(text):
#     doc = nlp(text)
#     lemmatized_text = [token.lemma_ for token in doc if token.text.lower() not in spacy_stop_words]
#     return ' '.join(lemmatized_text)

# df = pd.read_csv("sentimentdataset (Project 1).csv")

# df.drop(columns = ['Source', 'ID'], inplace=True)

# # Apply the function to the "Message" column
# df['Message'] = df['Message'].apply(lemmatize_and_remove_stop_words)

# print(df['Message'])

# Save the updated DataFrame to a new CSV file
# df.to_csv('sentimentdataset_stopwords_lemmatized.csv', index=False)


df = pd.read_csv("sentimentdataset_stopwords_lemmatized.csv")


X = df['Message']
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transform the training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf)

# to convert it to a data frame
# df_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

svc = LinearSVC(max_iter=1000000, dual=False)

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'tol': [1e-4, 1e-3, 1e-2],
    'class_weight': [None, 'balanced']
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters found by grid search
print("Best Parameters: ", grid_search.best_params_)

# Predict on the testing set with the best model
y_pred = grid_search.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Testing Set: {:.2f}%".format(accuracy * 100))

# Display additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))