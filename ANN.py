import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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
sentence_embeddings = vectorizer.fit_transform(X)

# print("\nTf-idf Feature Names:")
# print(vectorizer.get_feature_names_out())

# print("\nTf-idf Matrix:")
# print(sentence_embeddings.toarray())


X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings.toarray(), y, test_size=0.2, random_state=42)

def build_model(neurons=16, learning_rate=0.01):
  model = Sequential()
  model.add(layers.Dense(units=neurons, activation='relu', input_dim=X_train.shape[1]))
  model.add(layers.Dense(units=neurons, activation='relu'))
  model.add(layers.Dense(units=1, activation='sigmoid'))

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=build_model,neurons = [16, 32, 64, 128], learning_rate = [0.001, 0.01, 0.1], batch_size=[16, 32, 64], verbose=0)

param_grid = {
    'neurons': [16, 32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

print("Best: ANN model is %f using %s" % (grid_result.best_score_, grid_result.best_params_))
best_model = grid_result.best_estimator_
best_model.fit(X_train, y_train)
prediction = best_model.predict(X_test)
acc = accuracy_score(y_pred=prediction, y_true=y_test)
print(f'Test Accuracy: {acc}')

report = classification_report(y_test, prediction)
print("Classification Report:\n", report)