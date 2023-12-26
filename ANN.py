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

df = pd.read_csv("sentimentdataset_stopwords_lemmatized.csv")

features = ["Message"]
targets = ["Target"]


X = df.iloc[:, 0]
y = df.iloc[:, -1]

vectorizer = TfidfVectorizer()
sentence_embeddings = vectorizer.fit_transform(X)

# print("\nTf-idf Feature Names:")
# print(vectorizer.get_feature_names_out())

# print("\nTf-idf Matrix:")
# print(sentence_embeddings.toarray())


X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, y, test_size=0.2, random_state=42)

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